"""

An end-to-end cloud-based Metaflow pipeline for book recommendations, 
starting from raw data and ending with a real-time prediction endpoint. 

The original dataset used is in the Public Domain and can be found on 
Kaggle: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

"""

from metaflow import FlowSpec, step, S3, Parameter, current, conda_base
import time
from collections import defaultdict

class BookRecSysFlow(FlowSpec):

    @step
    def start(self):
        """
        Start the flow
        """
        from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        print("datastore is: %s" % DATASTORE_SYSROOT_S3)
        self.next(self.prepare_dataset)
        
        

    def filter_dataset(self, dataset, min_book_ratings=20, min_user_ratings=20):
        """
        Filters the dataset based on minimum book and user ratings.

        Args:
            dataset (DataFrame): The input dataset containing book ratings.
            min_book_ratings (int, optional): Minimum number of ratings a 
                                              book should have to be included. 
                                              Default is 20.
            min_user_ratings (int, optional): Minimum number of ratings a 
                                              user should have to be included. 
                                              Default is 20.

        Returns:
            DataFrame: Filtered dataset containing only books and users with 
                       sufficient ratings.
        """
        # Filter books with at least min_book_ratings ratings
        book_counts = dataset['ISBN'].value_counts()
        filter_books = book_counts[book_counts > min_book_ratings].index.tolist()

        # Filter users with at least min_user_ratings ratings
        user_counts = dataset['User-ID'].value_counts()
        filter_users = user_counts[user_counts > min_user_ratings].index.tolist()

        # Filter the dataset based on the filtered books and users
        dataset_filtered = dataset[
            dataset['ISBN'].isin(filter_books) &
            dataset['User-ID'].isin(filter_users)
        ]

        # Print original and new data frame shapes
        print(f'# The original data frame shape:\t{dataset.shape}')
        print(f'# The new data frame shape:\t{dataset_filtered.shape}')

        return dataset_filtered
    
    @step
    def prepare_dataset(self):
        """
        Prepares the dataset and defines the models for grid search training.

        Reads and merges the books, users, and ratings data.
        Filters the dataset based on minimum book and user ratings.
        Creates a Surprise Dataset from the filtered data.
        Sets up the models and their hyperparameter sets for grid search training.

        """
        import pandas as pd
        from surprise import Dataset, Reader
        from surprise import BaselineOnly, KNNBasic, NMF, SVD

        # Read and load the books, users, and ratings data
        books = pd.read_csv('../Data/Books.csv', delimiter = ",", usecols = [0,1,2,3,4], on_bad_lines='skip')
        users = pd.read_csv('../Data/Users.csv')
        ratings = pd.read_csv('../Data/Ratings.csv')

        self.books = books
        self.users = users
        self.ratings = ratings
        self.books_dict = pd.Series(books['Book-Title'].values, index=books['ISBN']).to_dict()

        # Merge users and ratings to create the main dataset
        dataset = pd.merge(users, ratings, on='User-ID', how='inner')

        # Filter the dataset based on minimum book and user ratings
        dataset = self.filter_dataset(dataset)
        self.dataset = dataset

        # Create Surprise Dataset from the filtered data
        reader = Reader(rating_scale=(0, 9))
        data = Dataset.load_from_df(self.dataset[['User-ID', 'ISBN', 'Book-Rating']], reader)
        self.data = data
        
        # Define sets of models and their corresponding hyperparameter options for grid search training
        self.models_sets = [
            (
                "baseline",
                BaselineOnly,
                {'bsl_options': {'method': ['als', 'sgd'],'reg': [0.02, 0.05, 0.1]}}
            ),
            (
                "knn",
                KNNBasic, 
                {'k': [10, 20],'min_k': [3, 5]}
            ),
            (
                "nmf",
                NMF, 
                {'n_factors': [50, 100],'n_epochs': [20, 50]}
            ),
            (
                "svd",
                SVD, 
                {'n_factors': [50, 100], 'n_epochs': [20, 50], 
                'lr_all': [0.002, 0.01], 'reg_all': [0.02, 0.5]}
            )
        ]

        # Proceed to grid search training step for each model set
        self.next(self.gridsearch_training, foreach="models_sets")

    # https://outerbounds.com/docs/nested-foreach/
    @step
    def gridsearch_training(self):
        """
        Performs grid search for hyperparameter tuning and trains the model.

        Uses Surprise's GridSearchCV to find the best hyperparameters for the model.
        Trains the model with the best hyperparameters on the filtered dataset.

        """
        from surprise.model_selection import GridSearchCV

        # Unpack the input containing the model name, model class, and parameter dictionary
        self.model_name, self.model, self.param_dict = self.input

        print(type(self.param_dict))

        # Perform grid search with cross-validation for RMSE as the evaluation metric
        grid_search = GridSearchCV(self.model, self.param_dict, measures=['rmse'], cv=3)
        grid_search.fit(self.data)

        print('Model trained')

        # Retrieve the best hyperparameters and corresponding score
        self.params = grid_search.best_params['rmse']
        self.scores = grid_search.best_score['rmse']

        # Create the model instance with the best hyperparameters
        self.clf = self.model(**self.params)

        # Proceed to the next step for model training
        self.next(self.join_train)

    @step
    def join_train(self, inputs):
        """
        Joins the training results from multiple models.

        Combines the model names, trained models, and mean accuracy scores from inputs.
        Selects the best model based on mean accuracy score.

        Args:
            inputs (List): A list of inputs containing model training results.

        """
        import json

        # Initialize a dictionary to store the model training results
        self.scores = {
            'model_name': [],
            'model': [],
            'params': [],
            'mean accuracy': []
        }

        # Collect model names, trained models, and mean accuracy scores from inputs
        for i in inputs:
            self.scores['model_name'].append(i.model_name)
            self.scores['model'].append(i.clf)
            self.scores['mean accuracy'].append(i.scores)
            
        # Store the dataset from the first input (assuming all inputs use the same dataset)
        self.data = inputs[0].data

        # Proceed to the next step for selecting the best model
        self.next(self.select_best_model)

    # From there https://surprise.readthedocs.io/en/stable/FAQ.html
    def precision_recall_at_k(self, predictions, k=10, threshold=3.5):
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls

    @step
    def select_best_model(self):
        """
        Selects the best model based on precision and recall scores.

        Fits each model with the training dataset and computes precision and recall at k=5, threshold=4.
        Chooses the best model based on precision and recall scores.
        Sets the selected model and its corresponding predictions for the next step.

        """
        self.trainset = self.data.build_full_trainset()
        self.testset = self.trainset.build_testset()  
        
        results = defaultdict(dict)
        predictions_ = defaultdict()

        for name, algo in zip(self.scores['model_name'], self.scores['model']):
            algo.fit(self.trainset)
            predictions = algo.test(self.testset)
            precisions, recalls = self.precision_recall_at_k(predictions, k=5, threshold=4)

            average_precision = sum(prec for prec in precisions.values()) / len(precisions)
            average_recall = sum(rec for rec in recalls.values()) / len(recalls)

            results[name] = {'precision@k': average_precision, 'recall@k': average_recall}
            predictions_[name] = predictions

            print(f'''Precision {name}: {average_precision}
                    Recall {name}: {average_recall}
                    ''')

        best_by_precision = sorted(results, key=lambda x: results[x]['precision@k'])
        best_by_recall = sorted(results, key=lambda x: results[x]['recall@k'])

        best_model_by_precision = best_by_precision[-1]
        best_model_by_recall = best_by_recall[-1]

        print(f'''The best model by precision is {best_model_by_precision} 
            and the best model by recall is {best_model_by_recall}.)''')

        index_of_best_model = self.scores['model_name'].index(best_model_by_recall)
        model_selected = self.scores['model'][index_of_best_model]
        self.model = model_selected

        predictions_ = predictions_[best_model_by_recall]
        self.predictions = predictions_

        # Proceed to the next step for building the retrieval model
        self.next(self.build_retrieval_model)

    # From there https://surprise.readthedocs.io/en/stable/FAQ.html
    def get_top_n(self, predictions, n=10):
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n
    
    @step
    def build_retrieval_model(self):
        """
        Builds the retrieval model and ensures its consistency.

        Dumps the trained algorithm and reloads it to generate a signature for the endpoint with a timestamp.
        Verifies that the reloaded algorithm produces the same predictions as the original algorithm.
        Proceeds to the final step.

        """
        from surprise import dump

        # Dump algorithm and reload it.
        # Generate a signature for the endpoint and timestamp as a convention
        self.model_timestamp = int(round(time.time() * 1000))
        local_name = "./static/model-{}.pkl".format(self.model_timestamp)
        dump.dump(local_name, algo=self.model)
        _, loaded_algo = dump.load(local_name)

        # We now ensure that the algo is still the same by checking the predictions.
        predictions_loaded_algo = loaded_algo.test(self.testset)
        assert self.predictions == predictions_loaded_algo

        # Proceed to the final step
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow
        """
        return

if __name__ == "__main__":
    BookRecSysFlow()