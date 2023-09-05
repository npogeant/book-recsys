# python model_flow.py run --ratings "[(242, '0553278223', 10.0)]"

import ast
from metaflow import FlowSpec, step, Flow, Parameter
 
class BookRecSysPredictFlow(FlowSpec):
 
    ratings = Parameter('ratings', required=True)
 
    @step
    def start(self):
        run = Flow('BookRecSysFlow').latest_run  
        self.train_run_id = run.pathspec   
        self.model = run['end'].task.data.model 
        run = Flow('BookRecSysFlow').latest_run  
        self.train_run_id = run.pathspec   
        self.model = run['end'].task.data.model
        self.int_data = ast.literal_eval(self.ratings)
        print("Input ratings", self.int_data)
        self.next(self.predict)
        
    @step
    def predict(self):
        self.results = self.model.test(self.int_data)
        self.next(self.end)
        
    
 
    @step
    def end(self):
        print('Results', self.results)
 
if __name__ == '__main__':
     BookRecSysPredictFlow()