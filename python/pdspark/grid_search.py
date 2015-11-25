"""
Class for parallelizing GridSearchCV jobs in scikit-learn
"""

from collections import namedtuple

import numpy as np
import scipy
from scipy.sparse.csr import csr_matrix

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV as SKL_GridSearchCV


class GridSearchCV(object):
    """
    Class for parallelizing GridSearchCV jobs in scikit-learn.

    This resembles sklearn.grid_search.GridSearchCV, with some parameters fixed:
     - iid = False
     - refit = True
    """

    def __init__(self, sc, estimator, param_grid, **params):
        """
        __init__(self, sc, estimator, param_grid, cv=3, random_state=12345)
        :param sc: SparkContext
        :param estimator: scikit-learn estimator, such as a Pipeline
        :param param_grid: Grid of parameters, in scikit-learn format
        """
        self.sc = sc
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = 3
        self.random_state = 12345
        for p in params:
            if p == 'cv':
                self.cv = int(params[p])
            elif p == 'random_state':
                self.random_state = int(params[p])
            else:
                raise KeyError("GridSearchCV was given unrecognized param %s (with value %s)" %
                               (p, params[p]))
        self.sklearn_model_ = None

    def fit(self, X, y):
        """
        This resembles scikit-learn's GridSearchCV.fit.
        This returns a scikit-learn instance with the fitted model, and it stores that instance
        in this class as `sklearn_model`.

        :param X: Features matrix of size number of instances (rows) x number of features (cols)
        :param y: Labels column vector of length number of instances (rows)
        :return: Fitted model, as a fitted sklearn.grid_search.GridSearchCV instance.
                 Note: The scikit-learn instance's field `best_score_` is set to the score of
                 the final refit model on all training data.
        """
        if isinstance(X, np.ndarray) and len(X) > 0 and\
                isinstance(X[0], csr_matrix) and X[0].shape[0] == 1:
            # concatenate rows into a single csr_matrix
            X = scipy.sparse.vstack(X).tocsr()

        param_grid = self._getSparkParamGrid(self.param_grid)
        # zip params with indices for reconstructing later
        param_fold_grid = [(i, param_grid[i], fold)
                           for fold in range(self.cv)
                           for i in range(len(param_grid))]

        par_param_fold_grid = self.sc.parallelize(param_fold_grid, len(param_fold_grid))
        X_bc = self.sc.broadcast(X)
        y_bc = self.sc.broadcast(y)

        estimator = self.estimator
        cv = self.cv
        random_state = self.random_state
        trainFold = self._trainFold
        allResults = par_param_fold_grid.map(
            lambda (params_i, params, fold): (params_i, trainFold(estimator, X_bc, y_bc, fold,
                                                                  params, cv, random_state)))\
            .collect()
        X_bc.unpersist()
        y_bc.unpersist()
        grid_scores = [[] for _ in range(len(param_grid))]
        for (params_i, (params, (fold, metric))) in allResults:
            grid_scores[params_i].append((fold, metric))
        grid_scores = [GridSearchCV.GridScore.create(param_grid[i], grid_scores[i])
                       for i in range(len(grid_scores))]
        best_grid_score = max(grid_scores, key=lambda gs: gs.mean_validation_score)

        # Refit on best parameters
        best_estimator = self.estimator
        best_params = best_grid_score.parameters
        best_estimator.set_params(**best_params)
        best_estimator.fit(X, y)
        best_score = best_estimator.score(X, y)

        # Create scikit-learn GridSearchCV
        skl_gs =\
            SKL_GridSearchCV(self.estimator, self.param_grid, iid=False, refit=True, cv=self.cv)
        skl_gs.grid_scores_ = grid_scores
        skl_gs.best_estimator_ = best_estimator
        # skl_gs.best_score_  # How is this set if there is no held-out data after retraining???
        skl_gs.best_params_ = best_params
        skl_gs.scorer_ = best_estimator.score
        skl_gs.best_score_ = best_score

        self.sklearn_model_ = skl_gs
        return self.sklearn_model_

    @property
    def best_score_(self):
        if self.sklearn_model_ is not None:
            return self.sklearn_model_.best_score_
        else:
            raise RuntimeError("GridSearchCV must be fit before best_score_ field is available")

    @property
    def best_estimator_(self):
        if self.sklearn_model_ is not None:
            return self.sklearn_model_.best_estimator_
        else:
            raise RuntimeError("GridSearchCV must be fit before best_estimator_ field is available")

    class GridScore(namedtuple("GridScore",
                               ["parameters", "mean_validation_score", "cv_validation_scores"])):
        """
        Score results for a single parameter setting (point in the parameter grid).
        Named tuple with fields:
         - parameters: dict of params
         - mean_validation_score: mean score over the folds of CV
         - cv_validation_scores: list of scores for each fold of CV
        """

        @staticmethod
        def create(params, results):
            """
            :param params  parameters field
            :param results  list of results from _trainFold, each having (fold, metric)
            """
            return GridSearchCV.GridScore(params,
                                          np.mean([r[1] for r in results]),
                                          map(lambda r: r[1], sorted(results)))


    @staticmethod
    def _trainFold(estimator, X_bc, y_bc, fold, params, nfolds, random_state):
        """
        Train model for one (fold, point in param grid).
        :param estimator:
        :param X_bc:
        :param y_bc:
        :param fold:
        :param params:
        :param nfolds:
        :param random_state:
        :return: (params, (fold, metric))
        """
        local_X = X_bc.value
        local_y = y_bc.value
        if hasattr(local_X, 'shape'):
            n = local_X.shape[0]
        else:
            n = len(local_X)
        kfold = KFold(n=n, n_folds=nfolds, random_state=random_state)
        folds = [f for f in kfold]
        train_index, test_index = folds[fold]
        # Set parameters
        estimator.set_params(**params)
        # Train model
        estimator.fit(local_X[train_index], local_y[train_index])
        # Evaluate model
        metric = estimator.score(local_X[test_index], local_y[test_index])
        return (params, (fold, metric))

    def _getSparkParamGrid(self, param_grid):
        """
        Given a scikit-learn parameter grid, create a list of parameter sets analogous to a
        list of MLlib ParamMaps.
        :param param_grid: Either a dict mapping param -> list of param values,
                           or a list of such dicts.  Tuples are treated like lists.
        :return: List of dicts, where each dict can be converted into keyword arguments
                 to pass to a scikit-learn estimator fit() method.
        """
        if isinstance(param_grid, dict):
            grid = [[]]
            for p in param_grid:
                vals = param_grid[p]
                if isinstance(vals, tuple):
                    vals = list(vals)
                elif not isinstance(vals, list):
                    raise TypeError("param_grid must be a dict of param->[param vals],"
                                    " or a list of dicts.  Found a dict, but key %s had value of"
                                    " type %s" % (p, type(vals)))
                grid = [g + [(p,v)] for v in vals for g in grid]
            return [dict(g) for g in grid]
        elif isinstance(param_grid, list):
            grid = []
            for d in param_grid:
                if not isinstance(d, dict):
                    raise TypeError("param_grid must be a dict of param->[param vals],"
                                    " or a list of dicts.  Found a list with element of"
                                    " type: %s" % type(d))
                g = self._getSparkParamGrid(d)
                grid += g
            return grid
        elif isinstance(param_grid, tuple):
            return self._getSparkParamGrid(list(param_grid))
        else:
            raise TypeError("param_grid must be a dict of param->[param vals],"
                            " or a list of dicts.  Found type: %s" % type(self.param_grid))

    @property
    def sklearn_model(self):
        if self.sklearn_model_ is not None:
            return self.sklearn_model_
        else:
            raise RuntimeError("GridSearchCV.sklearn_model will only exist after fit()"
                               " has been called")
