
from itertools import chain, repeat, cycle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import unittest

from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
import sklearn.base

from spark_sklearn.keyed_models import KeyedEstimator, KeyedModel, SparkSklearnEstimator
from spark_sklearn.test_utils import fixtureReuseSparkSession, assertPandasAlmostEqual, RandomTest

def _sortByComponentWeight(pca):
    zipped = zip(pca.components_, pca.explained_variance_ratio_)
    ordered = sorted(zipped, key=lambda x: x[1])
    return tuple(np.array(unzipped) for unzipped in zip(*ordered))

def _assertPandasAlmostEqual(actual, expected, sortby):
    def convert_estimators(x): # note convertion makes estimators invariant to training order.
        if isinstance(x, SparkSklearnEstimator):
            x = x.estimator
        if isinstance(x, LinearRegression) or isinstance(x, LogisticRegression):
            return x.coef_, x.intercept_
        if isinstance(x, PCA):
            return _sortByComponentWeight(x)
        if isinstance(x, KMeans):
            return x.cluster_centers_, x.labels_
        return x
    assertPandasAlmostEqual(actual, expected, convert=convert_estimators, sortby=sortby)

@fixtureReuseSparkSession
class KeyedModelTests(RandomTest):

    NDIM = 5

    class _CustomClusterer(sklearn.base.BaseEstimator):
        def fit(X, y=None):
            pass
        def transform(X):
            return X
        def fit_predict(X):
            return np.zeros(len(X))

    class _CustomTransformer(sklearn.base.BaseEstimator):
        def fit(X): # Only 1 argument expected!
            pass
        def transform(X):
            return X
        def predict(X): # Dummy predict to throw us off - all sklearn clusterers have fit_predict
            return np.zeros(len(X))

    class _CustomMissingFit(sklearn.base.BaseEstimator):
        def transform(X):
            return X

    # Makes sure that that the parameter estimator and its generated model are of the given type.
    # Simultaneously makes sure that an empty fit() works.
    def checkEstimatorType(self, keyedEstimator, expectedType):
        self.assertEqual(keyedEstimator.sklearnEstimatorType, expectedType)
        schema = StructType().add("features", DoubleType()).add("key", LongType())
        yCol = keyedEstimator.getOrDefault("yCol")
        if yCol is not None:
            schema = schema.add(yCol, DoubleType())
        emptyDF = self.spark.createDataFrame([], schema=schema)
        keyedModel = keyedEstimator.fit(emptyDF)
        self.assertEqual(keyedModel.sklearnEstimatorType, expectedType)

    def test_correct_estimator_type(self):
        self.checkEstimatorType(KeyedEstimator(sklearnEstimator=PCA()), "transformer")

        self.checkEstimatorType(KeyedEstimator(sklearnEstimator=LinearRegression(), yCol="y"),
                                "predictor")

        self.checkEstimatorType(KeyedEstimator(sklearnEstimator=DBSCAN()), "clusterer")

        self.checkEstimatorType(KeyedEstimator(sklearnEstimator=KMeans()), "clusterer")

        ke = KeyedEstimator(sklearnEstimator=KMeans(), estimatorType="transformer")
        self.checkEstimatorType(ke, "transformer")

        custom = KeyedModelTests._CustomClusterer()
        ke = KeyedEstimator(sklearnEstimator=custom)
        self.checkEstimatorType(ke, "clusterer")

        ke = KeyedEstimator(sklearnEstimator=custom, estimatorType="transformer")
        self.checkEstimatorType(ke, "transformer")

        custom = KeyedModelTests._CustomTransformer()
        self.checkEstimatorType(KeyedEstimator(sklearnEstimator=custom), "transformer")

    def test_invalid_argument(self):
        # Need to specify sklearnEstimator
        self.assertRaises(ValueError, KeyedEstimator)

        # sklearnEstimator must be a sklearn.base.Estimator
        create = lambda: KeyedEstimator(sklearnEstimator=5)
        self.assertRaises(ValueError, create)
        class SomeUDC(object):
            pass
        create = lambda: KeyedEstimator(sklearnEstimator=SomeUDC())
        self.assertRaises(ValueError, create)

        # Must have fit()
        create = lambda: KeyedEstimator(sklearnEstimator=KeyedModelTests._CustomMissingFit())
        self.assertRaises(AttributeError, create)

        # Must have key columns
        create = lambda: KeyedEstimator(sklearnEstimator=PCA(), keyCols=[])
        self.assertRaises(ValueError, create)

        # Columns can't have "estimator" name in them
        create = lambda: KeyedEstimator(sklearnEstimator=PCA(), keyCols=["key", "estimator"])
        self.assertRaises(ValueError, create)
        create = lambda: KeyedEstimator(sklearnEstimator=PCA(), xCol="estimator")
        self.assertRaises(ValueError, create)
        create = lambda: KeyedEstimator(sklearnEstimator=LinearRegression(), yCol="estimator")
        self.assertRaises(ValueError, create)
        create = lambda: KeyedEstimator(sklearnEstimator=PCA(), yCol="estimator")
        self.assertRaises(ValueError, create)

        # Presence of yCol requires predictor
        create = lambda: KeyedEstimator(sklearnEstimator=LinearRegression(), yCol="y",
                                        estimatorType="transformer")
        self.assertRaises(ValueError, create)
        create = lambda: KeyedEstimator(sklearnEstimator=LinearRegression(), yCol="y",
                                        estimatorType="clusterer")
        self.assertRaises(ValueError, create)

        # estimatorType must be one of the three options
        create = lambda: KeyedEstimator(sklearnEstimator=PCA(), estimatorType="regressor")
        self.assertRaises(ValueError, create)

    # Checks that only the model throws an AttributeError at transform time.
    def checkPredictionAttrError(self, keyedEstimator):
        schema = StructType().add("features", DoubleType()).add("key", LongType())
        yCol = keyedEstimator.getOrDefault("yCol")
        if yCol is not None:
            schema = schema.add(yCol, DoubleType())
        emptyDF = self.spark.createDataFrame([], schema=schema)
        keyedModel = keyedEstimator.fit(emptyDF)
        self.assertRaises(AttributeError, keyedModel.transform, emptyDF)

    def test_attr_error(self):
        ke = KeyedEstimator(sklearnEstimator=PCA(), estimatorType="clusterer")
        self.checkPredictionAttrError(ke)
        ke = KeyedEstimator(sklearnEstimator=PCA(), yCol="y", estimatorType="predictor")
        self.checkPredictionAttrError(ke)

        ke = KeyedEstimator(sklearnEstimator=DBSCAN(), estimatorType="transformer")
        self.checkPredictionAttrError(ke)
        ke = KeyedEstimator(sklearnEstimator=DBSCAN(), yCol="y", estimatorType="predictor")
        self.checkPredictionAttrError(ke)

        # LinearRegression() or any other predictor would actually fail at fit-time if we used a
        # non-empty DF with the wrong estimatorType since no y value would be passed, so
        # scikit-learn would complain.

    def test_type_error(self):
        df = self.spark.createDataFrame([("a", 0), ("b", 0)]).toDF("features", "key")
        keyedPCA = KeyedEstimator(sklearnEstimator=PCA())
        self.assertRaises(TypeError, keyedPCA.fit, df)

        df = self.spark.createDataFrame([(Vectors.dense([i]), [i], 0) for i in range(10)])
        df = df.toDF("features", "y", "key")
        keyedLR = KeyedEstimator(sklearnEstimator=LinearRegression(), yCol="y")
        self.assertRaises(TypeError, keyedLR.fit, df)


    def checkKeyedModelEquivalent(self, minExamples, featureGen, labelGen, **kwargs):
        NUSERS = 10
        # featureGen() should generate a np rank-1 ndarray of equal length
        # labelGen() should generate a scalar
        assert (labelGen is not None) == ("yCol" in kwargs)
        isPredictor = labelGen is not None

        # sklearn's LinearRegression estimator is stable even if undetermined.
        # User keys are just [0, NUSERS), repeated for each key if there are multiple columns.
        # The i-th user has i examples.

        keyCols = kwargs.get("keyCols", KeyedEstimator._paramSpecs["keyCols"]["default"])
        outputCol = kwargs.get("outputCol", KeyedEstimator._paramSpecs["outputCol"]["default"])
        xCol = kwargs.get("xCol", KeyedEstimator._paramSpecs["xCol"]["default"])

        nExamplesPerUser = lambda i: max(minExamples, i + 1)
        userKeys = [[i for _ in keyCols] for i in range(NUSERS)]
        features = [[featureGen() for _ in range(nExamplesPerUser(i))] for i in range(NUSERS)]
        useless = [["useless col" for _ in range(nExamplesPerUser(i))] for i in range(NUSERS)]
        if isPredictor:
            labels = [[labelGen() for _ in range(nExamplesPerUser(i))] for i in range(NUSERS)]
        else:
            labels = None

        Xs = [np.vstack(x) for x in features]
        ys = [np.array(y) for y in labels] if isPredictor else repeat(None)
        localEstimators = [sklearn.base.clone(kwargs["sklearnEstimator"]).fit(X, y)
                           for X, y in zip(Xs, ys)]
        expectedDF = pd.DataFrame(userKeys, columns=keyCols)
        expectedDF["estimator"] = localEstimators

        def flattenAndConvertNumpy(x):
            return [Vectors.dense(i) if isinstance(i, np.ndarray) else i
                    for i in chain.from_iterable(x)]

        inputDF = pd.DataFrame.from_dict(
            {k: [i for i in range(NUSERS) for _ in range(nExamplesPerUser(i))] for k in keyCols})
        inputDF[xCol] = flattenAndConvertNumpy(features)
        inputDF["useless"] = flattenAndConvertNumpy(useless)
        if labels:
            inputDF[kwargs["yCol"]] = flattenAndConvertNumpy(labels)
        inputDF = self.spark.createDataFrame(inputDF)

        ke = KeyedEstimator(**kwargs)
        km = ke.fit(inputDF)

        actualDF = km.keyedModels.toPandas()
        _assertPandasAlmostEqual(actualDF, expectedDF, keyCols)

        # Test users with different amounts of points.
        nTestPerUser = lambda i: NUSERS // 4 if i < NUSERS // 2 else NUSERS * 3 // 4
        testFeatures = [[featureGen() for _ in range(nTestPerUser(i))] for i in range(NUSERS)]
        # "useless" column has nothing to do with computation, but is essential for keeping order
        # the same between the spark and non-spark versions
        useless = [range(nTestPerUser(i)) for i in range(NUSERS)]
        inputDF = pd.DataFrame.from_dict(
            {k: [i for i in range(NUSERS) for _ in range(nTestPerUser(i))] for k in keyCols})
        inputDF[xCol] = flattenAndConvertNumpy(testFeatures)
        inputDF["useless"] = flattenAndConvertNumpy(useless)

        estimatorType = km.sklearnEstimatorType # tested to be correct elsewhere
        def makeOutput(estimator, X):
            if estimatorType == "transformer":
                return estimator.transform(X)
            else:
                assert estimatorType == "predictor" or estimatorType == "clusterer"
                return estimator.predict(X).tolist()
        Xs = [np.vstack(x) for x in testFeatures]
        expectedOutput = map(makeOutput, localEstimators, Xs)
        expectedDF = inputDF.copy(deep=True)
        expectedDF[outputCol] = flattenAndConvertNumpy(expectedOutput)

        inputDF = self.spark.createDataFrame(inputDF)
        actualDF = km.transform(inputDF).toPandas()

        _assertPandasAlmostEqual(actualDF, expectedDF, keyCols + ["useless"])

    def test_transformer(self):
        minExamples = 1
        featureGen = lambda: np.random.random(KeyedModelTests.NDIM)
        labelGen = None
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=PCA())

    def test_clusterer(self):
        minExamples = 3
        featureGen = lambda: np.random.random(KeyedModelTests.NDIM)
        labelGen = None
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=KMeans(random_state=0,
                                                               n_clusters=minExamples))


    def test_regression_predictor(self):
        minExamples = 1
        featureGen = lambda: np.random.random(KeyedModelTests.NDIM)
        labelGen = lambda: np.random.random()
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=LinearRegression(), yCol="y")

    def test_classification_predictor(self):
        minExamples = 2
        featureGen = lambda: np.random.random(KeyedModelTests.NDIM)
        # Need to ensure each user has at least one of each label to train on.
        cyc = cycle([-1, 1])
        labelGen = lambda: next(cyc)
        lr = LogisticRegression(random_state=0)
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=lr, yCol="y")

    def test_diff_type_input(self):
        # Integer array
        minExamples = 1
        featureGen = lambda: np.random.randint(low=0, high=10, size=KeyedModelTests.NDIM)
        labelGen = lambda: np.random.random()
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=LinearRegression(), yCol="y")

        # float input
        featureGen = lambda: np.random.random()
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=LinearRegression(), yCol="y")

        # integer input
        featureGen = lambda: np.random.randint(100)
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=LinearRegression(), yCol="y")

    def test_no_defaults(self):
        minExamples = 1
        featureGen = lambda: np.random.random(KeyedModelTests.NDIM)
        labelGen = lambda: np.random.random()
        self.checkKeyedModelEquivalent(minExamples, featureGen, labelGen,
                                       sklearnEstimator=LinearRegression(), yCol="myy",
                                       xCol="myfeatures", keyCols=["mykey1", "mykey2"])

    def test_surprise_key(self):
        ke = KeyedEstimator(sklearnEstimator=PCA())
        schema = StructType().add("features", LongType()).add("key", LongType())
        df = self.spark.createDataFrame([], schema)
        km = ke.fit(df)

        self.assertEqual(km.keyedModels.collect(), [])
        self.assertEqual(km.keyedModels.dtypes,
                         [("key", LongType().simpleString()),
                          ("estimator", "sklearn-estimator")])

        df = self.spark.createDataFrame([(1, 2)], schema)
        df = km.transform(df)

        self.assertEqual(df.collect(), [(1, 2, None)])
        self.assertEqual(df.dtypes,
                         [("features", "bigint"),
                          ("key", "bigint"),
                          ("output", "vector")])
