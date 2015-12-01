__author__ = 'tjhunter'

import sklearn.grid_search
SKGridSearchCV = sklearn.grid_search.GridSearchCV

from pyspark import SparkContext
#from sklearn.grid_search import GridSearchCV as SKGridSearchCV
from pdspark import GridSearchCV2
# Overwrite the sklearn GridSearch in this suite so that we can run the same tests

def create_sc():
  return SparkContext('local[1]', "spark-sklearn tests")

sc = create_sc()

class SPGridSearchWrapper(GridSearchCV2):

  def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
    super(SPGridSearchWrapper, self).__init__(sc, estimator, param_grid, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score)

sklearn.grid_search.GridSearchCV = SPGridSearchWrapper
from sklearn.tests import test_grid_search
import unittest

# These methods do not raise ValueError but something different
blacklist = set(['test_pickle',
                 'test_grid_search_precomputed_kernel_error_nonsquare',
                 'test_grid_search_precomputed_kernel_error_kernel_function',
                 'test_grid_search_precomputed_kernel',
                 'test_grid_search_failing_classifier_raise'])

def all_methods():
  return [(mname, method) for (mname, method) in test_grid_search.__dict__.items()
          if mname.startswith("test_") and mname not in blacklist]

class AllTests(unittest.TestCase):
  pass

def create_method(method):
    def do_test_expected(*kwargs):
      method()
    return do_test_expected

for name, method in all_methods():
  test_method = create_method(method)
  test_method.__name__ = name
  setattr (AllTests, test_method.__name__, test_method)
