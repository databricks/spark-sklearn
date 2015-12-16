import unittest

import sklearn.grid_search

from spark_sklearn import GridSearchCV
from spark_sklearn.test_utils import create_sc
# Overwrite the sklearn GridSearch in this suite so that we can run the same tests with the same
# parameters.

sc = create_sc()

class SPGridSearchWrapper(GridSearchCV):

  def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
    super(SPGridSearchWrapper, self).__init__(sc, estimator, param_grid, scoring, fit_params,
            n_jobs, iid, refit, cv, verbose, pre_dispatch, error_score)

SKGridSearchCV = sklearn.grid_search.GridSearchCV
sklearn.grid_search.GridSearchCV = SPGridSearchWrapper
sklearn.grid_search.GridSearchCV_original = SKGridSearchCV
from sklearn.tests import test_grid_search

# These methods do not raise ValueError but something different
blacklist = set(['test_pickle',
                 'test_grid_search_precomputed_kernel_error_nonsquare',
                 'test_grid_search_precomputed_kernel_error_kernel_function',
                 'test_grid_search_precomputed_kernel',
                 'test_grid_search_failing_classifier_raise',
                 'test_grid_search_failing_classifier']) # This one we should investigate

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
