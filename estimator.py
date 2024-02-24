from typing import List, Tuple, Dict

class Estimator(object):
    def __init__(self, method_name):
        self.method_name = method_name

    def getMethod(self, method_name):
        '''
        `method_name` is a name of method we want to use.
        '''
        return self.switcher[method_name]

    def processData(self, X):
        method = self.getMethod(self.method_name)
        if isinstance(X, List):
            output = []
            for image in X:
                output.append(method(image))
        else: # If X is not List, it must be an image
            output = method(X)
        return output

    def forward(self, X):
        output = self.processData(X)
        return output
