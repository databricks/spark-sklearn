import unittest

import sklearn.grid_search
from spark_sklearn.grid_search import GridSearchCV
from spark_sklearn.random_search import RandomizedSearchCV
from spark_sklearn.test_utils import fixtureReuseSparkSession
# Overwrite the sklearn GridSearch in this suite so that we can run the same tests with the same
# parameters.


@fixtureReuseSparkSession
class AllTests(unittest.TestCase):
  
    # After testing, make sure to revert sklearn to normal (see _add_to_module())
    @classmethod
    def tearDownClass(cls):
        super(AllTests, cls).tearDownClass()
        # Restore sklearn module to the original state after done testing this fixture.
        sklearn.grid_search.GridSearchCV = sklearn.grid_search.GridSearchCV_original
        del sklearn.grid_search.GridSearchCV_original
        sklearn.grid_search.RandomizedSearchCV = sklearn.grid_search.RandomizedSearchCV_original
        del sklearn.grid_search.RandomizedSearchCV_original


class SPGridSearchWrapper(GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        super(SPGridSearchWrapper, self).__init__(AllTests.spark.sparkContext, estimator, param_grid,
                                                  scoring, fit_params, n_jobs, iid, refit, cv,
                                                  verbose, pre_dispatch, error_score)


class SPRandomizedSearchWrapper(RandomizedSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise'):
        super(SPRandomizedSearchWrapper, self).__init__(AllTests.spark.sparkContext, estimator,
                                                        param_distributions,
                                                        n_iter, scoring, fit_params, n_jobs, iid, refit, cv,
                                                        verbose,
                                                        pre_dispatch, random_state, error_score)


def _create_method(method):
    def do_test_expected(*_):
        method()
    return do_test_expected


def _add_to_module():
    # NOTE: This doesn't actually run scikit-learn tests against SPGridSearchWrapper
    # for scikit-learn >= 0.18, since the scikit-learn tests (in sklearn.model_selection.tests) use
    # sklearn.model_selection.GridSearchCV (not sklearn.grid_search.GridSearchCV)
    # TODO: Get scikit-learn tests to pass with spark-sklearn GridSearch implementation
    SKGridSearchCV = sklearn.grid_search.GridSearchCV
    sklearn.grid_search.GridSearchCV = SPGridSearchWrapper
    sklearn.grid_search.GridSearchCV_original = SKGridSearchCV

    SKRandomizedSearchCV = sklearn.grid_search.RandomizedSearchCV
    sklearn.grid_search.RandomizedSearchCV = SPRandomizedSearchWrapper
    sklearn.grid_search.RandomizedSearchCV_original = SKRandomizedSearchCV

    from sklearn.model_selection.tests import test_search
    all_methods = [(mname, method) for (mname, method) in test_search.__dict__.items()
                   if mname.startswith("test_")]

    for name, method in all_methods:
        method_for_test = _create_method(method)
        method_for_test.__name__ = name
        setattr(AllTests, method.__name__, method_for_test)


_add_to_module()
