"""
Class for parallelizing GridSearchCV jobs in scikit-learn
"""

from collections import namedtuple, Sized
import numpy as np
import scipy
from scipy.sparse.csr import csr_matrix

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.cross_validation import KFold, check_cv, _fit_and_score, _safe_split
from sklearn.grid_search import GridSearchCV as SKL_GridSearchCV
from sklearn.grid_search import BaseSearchCV, _check_param_grid, ParameterGrid, _CVScoreTuple
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable

class GridSearchCV2(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator, using Spark to
    distribute the computations.

    Important members are fit, predict.

    GridSearchCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation.

    Parameters
    ----------
    sc: the spark context

    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default 1
        This parameter is not used and kept for compatibility.

    pre_dispatch : int, or string, optional
        This parameter is not used and kept for compatibility.

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : integer or cross-validation generator, default=3
        A cross-validation generator to use. If int, determines
        the number of folds in StratifiedKFold if estimator is a classifier
        and the target y is binary or multiclass, or the number
        of folds in KFold otherwise.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

        The refitting step, if any, happens on the local machine.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.


    Examples
    --------
    >>> from sklearn import svm, grid_search, datasets
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = grid_search.GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape=None, degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params={}, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=...,
           scoring=..., verbose=...)


    Attributes
    ----------
    grid_scores_ : list of named tuples
        Contains scores for all parameter combinations in param_grid.
        Each entry corresponds to one parameter setting.
        Each named tuple has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    The parameters n_jobs and pre_dispatch are accepted but not used.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a an hyperparameter grid.

    :func:`sklearn.cross_validation.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """

    def __init__(self, sc, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        super(GridSearchCV2, self).__init__(
            estimator, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score)
        self.sc = sc
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, X, y=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        return self._fit(X, y, ParameterGrid(self.param_grid))

    def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        param_grid = [(parameters, train, test)
                      for parameters in parameter_iterable
                      for (train, test) in cv]
        # Because the original python code expects a certain order for the elements, we need to
        # respect it.
        indexed_param_grid = zip(range(len(param_grid)), param_grid)
        par_param_grid = self.sc.parallelize(indexed_param_grid)
        X_bc = self.sc.broadcast(X)
        y_bc = self.sc.broadcast(y)

        scorer = self.scorer_
        verbose = self.verbose
        fit_params = self.fit_params
        error_score = self.error_score
        fas = _fit_and_score

        def fun(tup):
            (index, (parameters, train, test)) = tup
            local_estimator = clone(base_estimator)
            local_X = X_bc.value
            local_y = y_bc.value
            res = fas(local_estimator, local_X, local_y, scorer, train, test, verbose,
                                  parameters, fit_params,
                                  return_parameters=True, error_score=error_score)
            return (index, res)
        indexed_out0 = dict(par_param_grid.map(fun).collect())
        out = [indexed_out0[idx] for idx in range(len(param_grid))]

        X_bc.unpersist()
        y_bc.unpersist()

        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, _, parameters in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


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
