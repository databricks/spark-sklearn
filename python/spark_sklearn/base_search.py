"""
Base class for parallelizing CV search jobs in scikit-learn
"""

from collections import defaultdict, Sized
from functools import partial
from itertools import islice
from random import randint
import numpy as np
from scipy.stats import rankdata

from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable


class SparkBaseSearchCV(BaseSearchCV):

    def __init__(self, estimator, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        super(SparkBaseSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _fit(self, X, y, groups, parameter_iterable):

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        # Some CV split implementations don't have random state, like LeaveOneOut.
        # Many do, like StratifiedKFold. This reseeds the random state on each call
        # if applicable.
        if hasattr(cv, 'random_state'):
            if not cv.random_state:
                cv.random_state = randint(1000, 9999)

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_param_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_param_candidates,
                                     n_param_candidates * n_splits))

        base_estimator = clone(self.estimator)

        param_grid = [(parameters, test_sequence_index)
                      for parameters in parameter_iterable
                      for test_sequence_index in range(n_splits)]
        # Because the original python code expects a certain order for the elements, we need to
        # respect it.
        indexed_param_grid = list(zip(range(len(param_grid)), param_grid))
        par_param_grid = self.sc.parallelize(indexed_param_grid, len(indexed_param_grid))
        X_bc = self.sc.broadcast(X)
        y_bc = self.sc.broadcast(y)
        groups_bc = self.sc.broadcast(groups)

        scorer = self.scorer_
        verbose = self.verbose
        error_score = self.error_score
        fit_params = self.fit_params
        return_train_score = self.return_train_score
        fas = _fit_and_score

        def fun(tup):
            (index, (parameters, test_sequence_index)) = tup
            local_estimator = clone(base_estimator)
            local_X = X_bc.value
            local_y = y_bc.value
            local_groups = groups_bc.value

            train, test = next(islice(
                cv.split(local_X, local_y, local_groups), test_sequence_index, test_sequence_index + 1))
            res = fas(local_estimator, local_X, local_y, scorer, train, test, verbose,
                      parameters, fit_params,
                      return_train_score=return_train_score,
                      return_n_test_samples=True, return_times=True,
                      return_parameters=True, error_score=error_score)
            return index, res
        indexed_out0 = dict(par_param_grid.map(fun).collect())
        out = [indexed_out0[idx] for idx in range(len(param_grid))]
        if return_train_score:
            (train_scores, test_scores, test_sample_counts, fit_time,
             score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts, fit_time, score_time, parameters) = zip(*out)
        X_bc.unpersist()
        y_bc.unpersist()
        groups_bc.unpersist()

        candidate_params = parameters[::n_splits]
        n_param_candidates = len(candidate_params)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            array = np.array(array, dtype=np.float64).reshape(n_param_candidates,
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
                                            np.empty(n_param_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
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
