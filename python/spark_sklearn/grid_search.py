"""
Class for parallelizing GridSearchCV jobs in scikit-learn
"""

import sys

from itertools import product
from collections import Sized, Mapping, namedtuple, defaultdict, Sequence
from functools import partial
import warnings
import numpy as np
from scipy.stats import rankdata

from sklearn.base import BaseEstimator, is_classifier, clone
#from sklearn.cross_validation import KFold, check_cv, _fit_and_score, _safe_split
from sklearn.model_selection import KFold, check_cv, ParameterGrid # new
from sklearn.model_selection._validation import _fit_and_score # new
from sklearn.utils.metaestimators import _safe_split  # new
#from sklearn.grid_search import BaseSearchCV, _check_param_grid, ParameterGrid, _CVScoreTuple
from sklearn.model_selection._search import BaseSearchCV, _check_param_grid, _CVScoreTuple # new
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import _num_samples, indexable

class GridSearchCV(BaseSearchCV):
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
    >>> from sklearn import svm, datasets
    >>> from spark_sklearn import GridSearchCV
    >>> from pyspark.sql import SparkSession
    >>> from spark_sklearn.util import createLocalSparkSession
    >>> spark = createLocalSparkSession()
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = GridSearchCV(spark.sparkContext, svr, parameters)
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
    >>> spark.stop(); SparkSession._instantiatedContext = None

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
                 pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True):
        super(GridSearchCV, self).__init__(
            estimator, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score, return_train_score)
        # super(GridSearchCV, self).__init__(
        #     estimator, scoring, fit_params, n_jobs, iid,
        #     refit, cv, verbose, pre_dispatch, error_score)
        self.sc = sc
        self.param_grid = param_grid
        # self.grid_scores_ = None
        self.cv_results_ = None # new
        _check_param_grid(param_grid)

    def fit_old(self, X, y=None):
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
        # print "Exiting"
        # sys.exit(0)
        return self._fit(X, y, ParameterGrid(self.param_grid))

    


    #ef _fit(self, X, y, parameter_iterable, groups=None):
    def fit(self, X, y=None, groups=None, **fit_params):

      if self.fit_params is not None:
        warnings.warn('"fit_params" as a constructor argument was '
                      'deprecated in version 0.19 and will be removed '
                      'in version 0.21. Pass fit parameters to the '
                      '"fit" method instead.', DeprecationWarning)
        if fit_params:
            warnings.warn('Ignoring fit_params passed as a constructor '
                          'argument in favor of keyword arguments to '
                          'the "fit" method.', RuntimeWarning)
        else:
            fit_params = self.fit_params

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        # Regenerate parameter iterable for each fit
        #candidate_params = list(self._get_param_iterator())
        #candidate_params = parameter_iterable # change later
        candidate_params = ParameterGrid(self.param_grid)
        n_candidates = len(candidate_params)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)

        param_grid = [(parameters, train, test) for parameters, (train, test) in product(candidate_params, cv.split(X, y, groups))]
        #print "PARAM GRID:",param_grid,"\n"
        #sys.exit(0)
        # Because the original python code expects a certain order for the elements, we need to
        # respect it.
        indexed_param_grid = list(zip(range(len(param_grid)), param_grid))
        par_param_grid = self.sc.parallelize(indexed_param_grid, len(indexed_param_grid))
        X_bc = self.sc.broadcast(X)
        y_bc = self.sc.broadcast(y)

        scorer = self.scorer_
        verbose = self.verbose
        #fit_params = self.fit_params # DEPRECIATED: remove later
        error_score = self.error_score
        return_train_score = self.return_train_score
        fas = _fit_and_score

        def fun(tup):
            (index, (parameters, train, test)) = tup
            local_estimator = clone(base_estimator)
            local_X = X_bc.value
            local_y = y_bc.value
            res = fas(local_estimator, local_X, local_y, scorer, train, test, verbose,
                                  parameters, fit_params,
                                  return_train_score=return_train_score, #was self.return_train_score (fixing this at true works??)
                                  return_n_test_samples=True, return_times=True,
                                  return_parameters=False, error_score=error_score)
            return (index, res)
        indexed_out0 = dict(par_param_grid.map(fun).collect())
        #print "Indexed out:",indexed_out0,"\n"
        out = [indexed_out0[idx] for idx in range(len(param_grid))]
        if return_train_score:
            (train_scores, test_scores, test_sample_counts, fit_time,
             score_time) = zip(*out)
        else:
            (test_scores, test_sample_counts, fit_time, score_time) = zip(*out)
        #print "TRAIN SCORES:",train_scores
        #print "SCORE TIME:",score_time

        # print "OUT:",out,"\n"
        # print "OUT[0]:",out[0]
        # print "OUT[0].keys:",out[0].keys()
        # sys.exit(0)
        X_bc.unpersist()
        y_bc.unpersist()
        #print "GOT HERE?!?!?!? - shouldn't happen"



        # pre_dispatch = self.pre_dispatch

        # out = Parallel(
        #     n_jobs=self.n_jobs, verbose=self.verbose,
        #     pre_dispatch=pre_dispatch
        # )(delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
        #                           train, test, self.verbose, parameters,
        #                           fit_params=fit_params,
        #                           return_train_score=self.return_train_score,
        #                           return_n_test_samples=True,
        #                           return_times=True, return_parameters=False,
        #                           error_score=self.error_score)
        #   for parameters, (train, test) in product(candidate_params,
        #                                            cv.split(X, y, groups)))

        # # if one choose to see train score, "out" will contain train score info
        # if self.return_train_score:
        #     (train_scores, test_scores, test_sample_counts, fit_time,
        #      score_time) = zip(*out)
        # else:
        #     (test_scores, test_sample_counts, fit_time, score_time) = zip(*out)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **fit_params)
            else:
                best_estimator.fit(X, **fit_params)
            self.best_estimator_ = best_estimator
        return self




    # def _fit_original(self, X, y, parameter_iterable):
    #     """Actual fitting,  performing the search over parameters."""

    #     estimator = self.estimator
    #     cv = self.cv
    #     self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

    #     n_samples = _num_samples(X)
    #     X, y = indexable(X, y)

    #     if y is not None:
    #         if len(y) != n_samples:
    #             raise ValueError('Target variable (y) has a different number '
    #                              'of samples (%i) than data (X: %i samples)'
    #                              % (len(y), n_samples))
    #     cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    #     #cv = check_cv(cv, y, classifier=is_classifier(estimator))

    #     if self.verbose > 0:
    #         if isinstance(parameter_iterable, Sized):
    #             n_candidates = len(parameter_iterable)
    #             print("Fitting {0} folds for each of {1} candidates, totalling"
    #                   " {2} fits".format(len(cv), n_candidates,
    #                                      n_candidates * len(cv)))

    #     base_estimator = clone(self.estimator)

    #     param_grid = [(parameters, train, test)
    #                   for parameters in parameter_iterable
    #                   for (train, test) in cv]
    #     # Because the original python code expects a certain order for the elements, we need to
    #     # respect it.
    #     indexed_param_grid = list(zip(range(len(param_grid)), param_grid))
    #     par_param_grid = self.sc.parallelize(indexed_param_grid, len(indexed_param_grid))
    #     X_bc = self.sc.broadcast(X)
    #     y_bc = self.sc.broadcast(y)

    #     scorer = self.scorer_
    #     verbose = self.verbose
    #     fit_params = self.fit_params
    #     error_score = self.error_score
    #     fas = _fit_and_score

    #     def fun(tup):
    #         (index, (parameters, train, test)) = tup
    #         local_estimator = clone(base_estimator)
    #         local_X = X_bc.value
    #         local_y = y_bc.value
    #         res = fas(local_estimator, local_X, local_y, scorer, train, test, verbose,
    #                               parameters, fit_params,
    #                               return_parameters=True, error_score=error_score)
    #         return (index, res)
    #     indexed_out0 = dict(par_param_grid.map(fun).collect())
    #     out = [indexed_out0[idx] for idx in range(len(param_grid))]
    #     # print "OUT:",out
    #     # print "OUT[0]:",out[0]
    #     # print "OUT[0].keys:",out[0].keys()
    #     # sys.exit(0)
    #     X_bc.unpersist()
    #     y_bc.unpersist()

    #     # Out is a list of triplet: score, estimator, n_test_samples
    #     n_fits = len(out)
    #     n_folds = len(cv)

    #     scores = list()
    #     grid_scores = list()
    #     for grid_start in range(0, n_fits, n_folds):
    #         n_test_samples = 0
    #         score = 0
    #         all_scores = []
    #         for this_score, this_n_test_samples, _, parameters in \
    #                 out[grid_start:grid_start + n_folds]:
    #             all_scores.append(this_score)
    #             if self.iid:
    #                 this_score *= this_n_test_samples
    #                 n_test_samples += this_n_test_samples
    #             score += this_score
    #         if self.iid:
    #             score /= float(n_test_samples)
    #         else:
    #             score /= float(n_folds)
    #         scores.append((score, parameters))
    #         # TODO: shall we also store the test_fold_sizes?
    #         grid_scores.append(_CVScoreTuple(
    #             parameters,
    #             score,
    #             np.array(all_scores)))
    #     # Store the computed scores
    #     self.grid_scores_ = grid_scores

    #     # Find the best parameters by comparing on the mean validation score:
    #     # note that `sorted` is deterministic in the way it breaks ties
    #     best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
    #                   reverse=True)[0]
    #     self.best_params_ = best.parameters
    #     self.best_score_ = best.mean_validation_score

    #     if self.refit:
    #         # fit the best estimator using the entire dataset
    #         # clone first to work around broken estimators
    #         best_estimator = clone(base_estimator).set_params(
    #             **best.parameters)
    #         if y is not None:
    #             best_estimator.fit(X, y, **self.fit_params)
    #         else:
    #             best_estimator.fit(X, **self.fit_params)
    #         self.best_estimator_ = best_estimator
    #     return self





    # def _fit_old(self, X, y, parameter_iterable):
    #     """Actual fitting,  performing the search over parameters."""

    #     estimator = self.estimator
    #     cv = self.cv
    #     self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

    #     n_samples = _num_samples(X)
    #     X, y = indexable(X, y)

    #     if y is not None:
    #         if len(y) != n_samples:
    #             raise ValueError('Target variable (y) has a different number '
    #                              'of samples (%i) than data (X: %i samples)'
    #                              % (len(y), n_samples))
    #     # cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    #     cv = check_cv(cv, y, classifier=is_classifier(estimator))

    #     if self.verbose > 0:
    #         if isinstance(parameter_iterable, Sized):
    #             n_candidates = len(parameter_iterable)
    #             print("Fitting {0} folds for each of {1} candidates, totalling"
    #                   " {2} fits".format(len(cv), n_candidates,
    #                                      n_candidates * len(cv)))

    #     base_estimator = clone(self.estimator)

    #     param_grid = [(parameters, train, test)
    #                   for parameters in parameter_iterable
    #                   for (train, test) in cv]
    #     # Because the original python code expects a certain order for the elements, we need to
    #     # respect it.
    #     indexed_param_grid = list(zip(range(len(param_grid)), param_grid))
    #     par_param_grid = self.sc.parallelize(indexed_param_grid, len(indexed_param_grid))
    #     X_bc = self.sc.broadcast(X)
    #     y_bc = self.sc.broadcast(y)

    #     scorer = self.scorer_
    #     verbose = self.verbose
    #     fit_params = self.fit_params
    #     error_score = self.error_score
    #     fas = _fit_and_score

    #     def fun(tup):
    #         (index, (parameters, train, test)) = tup
    #         local_estimator = clone(base_estimator)
    #         local_X = X_bc.value
    #         local_y = y_bc.value
    #         res = fas(local_estimator, local_X, local_y, scorer, train, test, verbose,
    #                               parameters, fit_params,
    #                               return_train_score=self.return_train_score,
    #                               return_n_test_samples=True, return_times=True,
    #                               return_parameters=True, error_score=error_score)
    #         return (index, res)
    #     indexed_out0 = dict(par_param_grid.map(fun).collect())
    #     out = [indexed_out0[idx] for idx in range(len(param_grid))]
    #     print "OUT:",out
    #     print "OUT[0]:",out[0]
    #     print "OUT[0].keys:",out[0].keys()
    #     sys.exit(0)
    #     X_bc.unpersist()
    #     y_bc.unpersist()

    #     # Out is a list of triplet: score, estimator, n_test_samples
    #     n_fits = len(out)
    #     n_folds = len(cv)

    #     scores = list()
    #     grid_scores = list()
    #     for grid_start in range(0, n_fits, n_folds):
    #         n_test_samples = 0
    #         score = 0
    #         all_scores = []
    #         for this_score, this_n_test_samples, _, parameters in \
    #                 out[grid_start:grid_start + n_folds]:
    #             all_scores.append(this_score)
    #             if self.iid:
    #                 this_score *= this_n_test_samples
    #                 n_test_samples += this_n_test_samples
    #             score += this_score
    #         if self.iid:
    #             score /= float(n_test_samples)
    #         else:
    #             score /= float(n_folds)
    #         scores.append((score, parameters))
    #         # TODO: shall we also store the test_fold_sizes?
    #         grid_scores.append(_CVScoreTuple(
    #             parameters,
    #             score,
    #             np.array(all_scores)))
    #     # Store the computed scores
    #     self.grid_scores_ = grid_scores

    #     # Find the best parameters by comparing on the mean validation score:
    #     # note that `sorted` is deterministic in the way it breaks ties
    #     best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
    #                   reverse=True)[0]
    #     self.best_params_ = best.parameters
    #     self.best_score_ = best.mean_validation_score

    #     if self.refit:
    #         # fit the best estimator using the entire dataset
    #         # clone first to work around broken estimators
    #         best_estimator = clone(base_estimator).set_params(
    #             **best.parameters)
    #         if y is not None:
    #             best_estimator.fit(X, y, **self.fit_params)
    #         else:
    #             best_estimator.fit(X, **self.fit_params)
    #         self.best_estimator_ = best_estimator
    #     return self
