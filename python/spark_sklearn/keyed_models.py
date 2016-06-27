"""
Classes for implementing keyed models.

The use case that this addresses is where a client has a dataset with many keys - the distribution
 of distinct keys amongst the is such that the total number of rows for every key value can be
contained completely in memory on a single machine.

This assumption is particularly enabling because clients may wish to apply more intricate single-
machine models (such as a scikit-learn estimator) to every user.

The API provided here generalizes the scikit-learn estimator interface to the Spark ML one; in
particular, it allows clients to train their scikit-learn estimators in parallel over a grouped
and aggregated dataframe.

>>> from sklearn.linear_model import LinearRegression
>>> from pyspark.ml.linalg import Vectors
>>> from pyspark.sql.functions import udf
>>> from pyspark.sql import SparkSession
>>> from spark_sklearn.util import createLocalSparkSession
>>> from spark_sklearn.keyed_models import KeyedEstimator
>>> spark = createLocalSparkSession()
>>> df = spark.createDataFrame([(user,
...                              Vectors.dense([i, i ** 2, i ** 3]),
...                              0.0 + user + i + 2 * i ** 2 + 3 * i ** 3)
...                             for user in range(3) for i in range(5)])
>>> df = df.toDF("key", "features", "y")
>>> df.where("5 < y and y < 10").sort("key", "y").show()
+---+-------------+---+
|key|     features|  y|
+---+-------------+---+
|  0|[1.0,1.0,1.0]|6.0|
|  1|[1.0,1.0,1.0]|7.0|
|  2|[1.0,1.0,1.0]|8.0|
+---+-------------+---+
<BLANKLINE>
>>> km = KeyedEstimator(sklearnEstimator=LinearRegression(), yCol="y").fit(df)
>>> def printFloat(x): return '{:.2f}'.format(round(x, 2))
>>> def printModel(model):
...     coef = "[" + ", ".join(map(printFloat, model.coef_)) + "]"
...     intercept = printFloat(model.intercept_)
...     return "intercept: {} coef: {}".format(intercept, coef)
...
>>> km.keyedModels.columns
['key', 'estimator']
>>> printedModels = udf(printModel)("estimator").alias("linear fit")
>>> km.keyedModels.select("key", printedModels).sort("key").show(truncate=False)
+---+----------------------------------------+
|key|linear fit                              |
+---+----------------------------------------+
|0  |intercept: 0.00 coef: [1.00, 2.00, 3.00]|
|1  |intercept: 1.00 coef: [1.00, 2.00, 3.00]|
|2  |intercept: 2.00 coef: [1.00, 2.00, 3.00]|
+---+----------------------------------------+
<BLANKLINE>

Now that we have generated a linear model for each key, we can apply it to keyed test data.
In the following, we only show one point for simplicity, but the test data can contain multiple
points for multiple different keys.

>>> input = spark.createDataFrame([(0, Vectors.dense(3, 1, -1))]).toDF("key", "features")
>>> km.transform(input).withColumn("output", udf(printFloat)("output")).show()
+---+--------------+------+
|key|      features|output|
+---+--------------+------+
|  0|[3.0,1.0,-1.0]|  2.00|
+---+--------------+------+
<BLANKLINE>
>>> spark.stop(); SparkSession._instantiatedContext = None # clear hidden SparkContext for reuse
"""

from itertools import chain
import numpy as np
import pickle
import sklearn.base
import sys

from pyspark import keyword_only
import pyspark.ml
from pyspark.ml.common import inherit_doc
from pyspark.ml.param import Param, Params
from pyspark.ml.linalg import Vector, Vectors
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import *
from pyspark.sql.types import _type_mappings, UserDefinedType, NumericType

from spark_sklearn.group_apply import gapply

class _SparkSklearnEstimatorUDT(UserDefinedType):
    """
    SQL user-defined type (UDT) for scikit-learn estimator
    """

    @classmethod
    def sqlType(cls):
        # In python 2, pickle serializes to strings, whereas in python 3, pickle uses bytes.
        return StringType() if sys.version_info[0] < 3 else BinaryType()

    @classmethod
    def module(cls):
        return "spark_sklearn.keyed_models"

    def serialize(self, obj):
        return pickle.dumps(obj.estimator)

    def deserialize(self, datum):
        return SparkSklearnEstimator(pickle.loads(datum))

    def simpleString(self):
        return "sklearn-estimator"

class SparkSklearnEstimator(object):
    """:class:`SparkSklearnEstimator` is a wrapper for containing scikit-learn estimators in
    dataframes - any estimators need to be stored inside the wrapper class to be properly
    serialized/deserialized in dataframe operations.

    Note any method called on the estimator this object wraps may be called on the wrapper instead.
    """

    __UDT__ = _SparkSklearnEstimatorUDT()

    def __init__(self, estimator):
        """Initializes with the parameter estimator.

        :param: estimator: scikit-learn estimator to contain.
        """
        self._estimator = estimator

    @property
    def estimator(self):
        """:return: the underlying estimator"""
        return self._estimator

    def __getattr__(self, item):
        if hasattr(self._estimator, item):
            return getattr(self._estimator, item)
        raise AttributeError()

def _isOneDimensional(schema, xCol):
    xType = schema[xCol].dataType
    return isinstance(xType, NumericType)

def _validateXCol(schema, xCol):
    if not _isOneDimensional(schema, xCol) and schema[xCol].dataType != Vector.__UDT__:
        raise TypeError("Input column {} is neither a numeric type nor a vector".format(xCol))

def _prepareXCol(series, is1D):
    if is1D:
        return series.values.reshape(-1, 1)
    else:
        return np.vstack(series.apply(lambda v: v.toArray()))

@inherit_doc
class KeyedEstimator(pyspark.ml.Estimator):
    """A :class:`KeyedEstimator` provides an interface for training per-key scikit-learn estimators.

    The :class:`KeyedEstimator` can be part of any Spark ML pipeline provided the columns are
    appropriately matched.

    Currently, the class provides a generalization for scikit-learn transformers and predictors
    (per-key clustering is currently unsupported). Because these scikit-learn estimators
    all derive from the same base type (yielding the same API), yet have different expectations
    for what methods should be called and with what arguments, this class enumerates two different
    types of behavior:

    1. ``"transformer"``

        Examples: ``sklearn.decomposition.PCA``, ``sklearn.cluster.KMeans``

        In this case, the estimator will aggregate the all input features for a given key into a
        `NxD` data matrix, where `N` is the number of rows with the given key and `D` is the
        feature space dimensionality; let this matrix be ``X``.

        For each such key and data matrix pair, a clone of the parameter estimator is fitted with
        ``estimator.fit(X)``, inducing a mapping between keys and fitted estimators: this produces
        a fitted transformer :class:`KeyedModel`, whose Spark ML ``transform()`` method generates an
        output column by applying each key's fitted scikit-learn estimator's own ``transform``
        method.

        The output column type for transformers will always be a :class:`DenseVector`.

    2. ``"predictor"``

        Examples: ``sklearn.svm.LinearSVC``, ``sklearn.linear_model.ElasticNet``

        Here, the estimator will likewise aggregate input features into the data matrix ``X``.
        In addition, the label column will be aggregated in a collated manner, generating
        a vector ``y`` for each key. The estimator clone will be fitted with
        ``estimator.fit(X, y)``.

        A predictor :class:`KeyedModel` transforms its input dataframe by generating an output
        column with the output of the estimator's ``predict`` method.

        The output column type for predictors will be the same as the label column (which
        must be an :class:`AtomicType` (else a :class:`TypeError` will be thrown at ``fit()``-time).

    The input column should be numeric or a vector (else a :class:`TypeError` will be thrown at
    ``fit()``-time). Don't use "estimator" as a column name.

    * Clustering is *not* supported.
      For instance, the scikit-learn ``KMeans`` estimator transforms its data into the
      cluster-mean-distance space - transform will not generate the column of ``KMeans``
      cluster labels.
    * Key-based grouping only occurs during training.
      During the transformation/prediction phase of computation, the output is unaggregated:
      the number of rows inputted as test data will be equal to the number of rows outputted.
    * ``spark.conf.get("spark.sql.retainGroupColumns")`` assumed to be ``u"true"``.
      This is the case by default for Spark 1.4+. This is necessary for both the keyed estimator
      and the keyed model.
    * Estimators trained, persisted, and loaded across different scikit-learn versions
      are not guaranteed to work.
    """

    _paramSpecs = {
        "sklearnEstimator": {"doc": "scikit-learn estimator applied to each group"},
        "keyCols": {"doc": "list of key column names", "default": ["key"]},
        "xCol": {"doc": "input features column name", "default": "features"},
        "outputCol": {"doc": "output column name", "default": "output"},
        "yCol": {"doc": "optional label column name", "default": None},
        "estimatorType": {"doc": "scikit-learn estimator type", "default": "transformer"}}

    @keyword_only
    def __init__(self, sklearnEstimator=None, keyCols=["key"], xCol="features",
                 outputCol="output", yCol=None):
        """For all instances, the ordered list of ``keyCols`` determine the set of groups which each
        ``sklearnEstimator`` is applied to.

        For every unique ``keyCols`` value, the remaining columns are aggregated and used to train the
        scikit-learn estimator.

        If ``yCol`` is specified, then this is assumed to be of ``"predictor"`` type, else a
        ``"transformer"``.

        :param sklearnEstimator: An instance of a scikit-learn estimator, with parameters configured
                                 as desired for each user.
        :param keyCols: Key column names list used to group data to which models are applied, where
                        order implies lexicographical importance.
        :param xCol: Name of column of input features used for training and
                     transformation/prediction.
        :param yCol: Specifies name of label column for regression or classification pipelines.
                     Required for predictors, must be unspecified or ``None`` for transformers.

        :raise ValueError: if ``sklearnEstimator`` is ``None``.
        :raise ValueError: if ``sklearnEstimator`` does not derive from
                           ``sklearn.base.BaseEstimator``.
        :raise ValueError: if ``keyCols`` is empty.
        :raise ValueError: if any column has the name ``"estimator"``
        :raise ValueError: if reflection checks indicate that parameter estimator is not equipped
                           with a ``fit()`` method or an appropriate transformation/prediction
                           method.
        """
        if sklearnEstimator is None:
            raise ValueError("sklearnEstimator should be specified")
        if not isinstance(sklearnEstimator, sklearn.base.BaseEstimator):
            raise ValueError("sklearnEstimator should be an sklearn.base.BaseEstimator")
        if len(keyCols) == 0:
            raise ValueError("keyCols should not be empty")
        if "estimator" in keyCols + [xCol, yCol]:
            raise ValueError("keyCols should not contain a column named \"estimator\"")

        # The superclass expects Param attributes to already be set, so we only init it after
        # doing so.
        for paramName, paramSpec in KeyedEstimator._paramSpecs.items():
            setattr(self, paramName, Param(Params._dummy(), paramName, paramSpec["doc"]))
        super(KeyedEstimator, self).__init__()
        self._setDefault(**{paramName: paramSpec["default"]
                            for paramName, paramSpec in KeyedEstimator._paramSpecs.items()
                            if "default" in paramSpec})
        kwargs = KeyedEstimator._inferredParams(self.__init__._input_kwargs)
        self._set(**kwargs)

        self._verifyEstimatorType()

    @staticmethod
    def _inferredParams(inputParams):
        if "yCol" in inputParams:
            inputParams["estimatorType"] = "predictor"
        return inputParams

    def _verifyEstimatorType(self):
        estimatorType = self.getOrDefault("estimatorType")
        estimator = self.getOrDefault("sklearnEstimator")

        if not hasattr(estimator, "fit"):
            raise ValueError("sklearnEstimator missing fit()")

        if estimatorType == "transformer":
            if not hasattr(estimator, "transform"):
                raise ValueError("estimatorType assumed to be transformer, but " +
                                 "sklearnEstimator is missing transform()")
        elif estimatorType == "predictor":
            if not hasattr(estimator, "predict"):
                raise ValueError("estimatorType assumed to be predictorr, but " +
                                 "sklearnEstimator is missing predict()")
        else:
            raise ValueError("estimatorType {} is unknown".format(estimatorType))

    def _fit(self, dataset):
        keyCols = self.getOrDefault("keyCols")
        xCol = self.getOrDefault("xCol")
        yCol = self.getOrDefault("yCol")
        isLabelled = yCol is not None
        estimatorType = self.getOrDefault("estimatorType")
        assert isLabelled == (estimatorType == "predictor"), \
            "yCol is {}, but it should {}be None for a {} estimatorType".format(
                yCol, "not " if isLabelled else "", estimatorType)

        _validateXCol(dataset.schema, xCol)

        cols = keyCols[:]
        cols.append(xCol)
        if isLabelled:
            cols.append(yCol)

        oneDimensional = _isOneDimensional(dataset.schema, xCol)
        projected = dataset.select(*cols) # also verifies all cols are present
        outputSchema = StructType().add("estimator", _SparkSklearnEstimatorUDT.sqlType())
        grouped = projected.groupBy(*keyCols)
        estimator = self.getOrDefault("sklearnEstimator")

        # Potential optimization: broadcast estimator

        import pandas as pd
        def fitEstimator(_, pandasDF):
            X = _prepareXCol(pandasDF[xCol], oneDimensional)
            y = pandasDF[yCol].values if isLabelled else None
            # Potential optimization - del pandasDF

            estimatorClone = sklearn.base.clone(estimator)
            estimatorClone.fit(X, y)
            pickled = pickle.dumps(estimatorClone)
            # Potential optimization - del estimatorClone

            # Until SPARK-15989 is resolved, we can't output the sklearn UDT directly here.
            return pd.DataFrame.from_records([(pickled,)])

        fitted = gapply(grouped, fitEstimator, outputSchema)

        extractSklearn = udf(lambda estimatorStr: SparkSklearnEstimator(pickle.loads(estimatorStr)),
                             SparkSklearnEstimator.__UDT__)
        keyedSklearnEstimators = fitted.select(
            *chain(keyCols, [extractSklearn(fitted["estimator"]).alias("estimator")]))

        if isLabelled:
            outputType = dataset.schema[yCol].dataType
        else:
            outputType = Vector.__UDT__

        return KeyedModel(keyCols=keyCols, xCol=xCol, outputCol=self.getOrDefault("outputCol"),
                          yCol=yCol, estimatorType=self.getOrDefault("estimatorType"),
                          keyedSklearnEstimators=keyedSklearnEstimators, outputType=outputType)

class KeyedModel(pyspark.ml.Model):
    """Represents a Spark ML Model, generated by a fitted :class:`KeyedEstimator`.

    Wraps fitted scikit-learn estimators - at transformation time transforms the
    input for each key using a key-specific model. See :class:`KeyedEstimator` documentation for
    details.

    If no estimator is present for a given key at transformation time, the prediction is null.
    """

    _paramSpecs = {
        "sklearnEstimator": {"doc": "sklearn estimator applied to each group"},
        "keyCols": {"doc": "list of key column names"},
        "xCol": {"doc": "input features column name"},
        "outputCol": {"doc": "output column name"},
        "yCol": {"doc": "optional label column name"},
        "estimatorType": {"doc": "scikit-learn estimator type"},
        "keyedSklearnEstimators": {"doc": "Dataframe of fitted sklearn estimators for each key"},
        "outputType": {"doc": "SQL type for output column"}}

    _sql_types = {v: k for k, v in _type_mappings.items()}

    @keyword_only
    def __init__(self, keyCols=None, xCol=None, outputCol=None, yCol=None, estimatorType=None,
                 keyedSklearnEstimators=None, outputType=None):
        """Used by :class:`KeyedEstimator` to generate a :class:`KeyedModel`. Not intended for
        external use."""

        assert (estimatorType == "predictor") == (yCol is not None), \
            "yCol is {}, but it should {}be None for a {} estimatorType".format(
                yCol, "not " if isLabelled else "", estimatorType)
        assert estimatorType == "transformer" or estimatorType == "predictor", estimatorType
        def implies(a, b):
            return not a or b
        assert implies(estimatorType == "transformer", outputType == Vector.__UDT__), outputType
        assert len(keyCols) > 0, len(keyCols)
        assert set(keyedSklearnEstimators.columns) == (set(keyCols) | set(["estimator"])), \
            "keyedSklearnEstimator columns {} should have both key columns {} and " + \
            "an estimator column".format(keyedSklearnEstimators.columns, keyCols)

        # The superclass expects Param attributes to already be set, so we only init it after
        # doing so.
        for paramName, paramSpec in KeyedModel._paramSpecs.items():
            setattr(self, paramName, Param(Params._dummy(), paramName, paramSpec["doc"]))
        super(KeyedModel, self).__init__()
        if yCol and type(outputType) not in KeyedModel._sql_types:
            raise TypeError("Output type {} is not an AtomicType (expected for {} estimator)"
                            .format(outputType, estimatorType))
        self._set(**self.__init__._input_kwargs)

    def _transform(self, dataset):
        keyCols = self.getOrDefault("keyCols")
        xCol = self.getOrDefault("xCol")
        outputCol = self.getOrDefault("outputCol")
        outputType = self.getOrDefault("outputType")

        _validateXCol(dataset.schema, xCol)

        # Potential optimization: group input data by key, then only deserialize each
        # estimator at most once. This becomes a bit difficult because then extraneous non-input
        # columns must be preserved, yet they can't just be collected into single rows since
        # we don't have a guarantee that they'll fit in memory.

        # Potential optimization: broadcast estimator

        shouldPredict = self.getOrDefault("estimatorType") == "predictor"
        oneDimensional = _isOneDimensional(dataset.schema, xCol)
        if shouldPredict:
            cast = KeyedModel._sql_types[type(outputType)]
        else:
            assert outputType == Vector.__UDT__, outputType
            # The closure of applyEstimator() doesn't know ahead of time that we won't use
            # the cast value, so it tries to serialize it.
            cast = None
        def applyEstimator(estimator, x):
            if not estimator:
                return None
            if oneDimensional:
                x = [[x]]
            else:
                x = x.toArray().reshape(1, -1)
            if shouldPredict:
                return cast(estimator.predict(x)[0])
            else:
                return Vectors.dense(estimator.transform(x)[0])
        transformation = udf(applyEstimator, outputType)

        joined = dataset.join(self.keyedModels, on=keyCols, how="left_outer")
        output = transformation(joined["estimator"], joined[xCol]).alias(outputCol)
        return joined.select(*chain(dataset, [output]))

    @property
    def keyedModels(self):
        """
        :return: Returns the ``keyedSklearnEstimators`` param, a :class:`DataFrame` with columns
                 ``keyCols`` (where each key is unique) and the column ``"estimator"`` containing
                 the fitted scikit-learn estimator as a :class:`SparkSklearnEstimator`.
        """
        return self.getOrDefault("keyedSklearnEstimators")
