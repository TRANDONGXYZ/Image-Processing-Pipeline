# import utils
# import numpy as np
# import multiprocessing

# class NormalizationPipeline(object):
#     def __init__(self, pipelines, workers=16, measure_time=False, verbose=False):
#         self.pipelines = pipelines
#         self.measure_time = measure_time
#         self.verbose = verbose
#         self.workers = workers

#     def _normalize_imgs(self, img):
#         for step in self.pipelines:
#             if type(step) is tuple:
#                 pipeline, argument = step
#             else:
#                 pipeline = step
#                 argument = {}

#             if (self.verbose):
#                 utils.printProgress(f'{pipeline.__name__} is running...')
#             img = pipeline(img, **argument)
#         return img
    
#     def forward(self, images):
#         if self.measure_time:
#             start = utils.getCurrentTime()
        
#         # Multiprocessing
#         # with multiprocessing.Pool(self.workers) as pool:
#         #     results = pool.map(self._normalize_imgs, images)

#         results = []
#         for img in images:
#             results.append(self._normalize_imgs(img))

#         if self.measure_time:
#             end = utils.getCurrentTime()
#             execution_time = np.round(end - start, decimals=3)
#             return results, execution_time
#         return results, None




from utils import Utils
import numpy as np
import multiprocessing

from estimator import Estimator

class NormalizationPipeline(Estimator):
    def __init__(self, pipeline, workers=16, gpu=False, measure_time=True, verbose=True):
        self.pipeline = pipeline
        self.gpu = gpu
        self.measure_time = measure_time
        self.verbose = verbose
        self.workers = workers

    def _getTupleOfStep(self, step):
        assert len(step) == 2, 'Size of tuple must be 2 (step_name, estimator).'
        step_name, est = step
        return step_name, est

    def forward(self, X):
        if self.measure_time:
            start = Utils.getCurrentTime()

        # Multiprocessing
        # with multiprocessing.Pool(self.workers) as pool:
        #     results = pool.map(self._normalize_imgs, images)
        
        output = X
        for step in self.pipeline:
            step_name, est = self._getTupleOfStep(step)
            if self.verbose:
                Utils.printMessage(f'---- Running {step_name} stage -----')
            output = est.forward(output)

        if self.measure_time:
            end = Utils.getCurrentTime()
            execution_time = round(end - start, 3)
            # Utils.printMessage(f'Execution time: {execution_time}s')
            return output, execution_time

        return output
