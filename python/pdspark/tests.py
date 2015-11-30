
import sys

if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest

import numpy as np
import pandas
import scipy
from scipy.sparse import csr_matrix

from sklearn.linear_model import Lasso as SKL_Lasso
from sklearn.linear_model import LogisticRegression as SKL_LogisticRegression
from sklearn.linear_model import LinearRegression as SKL_LinearRegression
from sklearn.feature_extraction.text import HashingVectorizer as SKL_HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as SKL_TfidfTransformer
from sklearn.grid_search import GridSearchCV as SKL_GridSearchCV
from sklearn.pipeline import Pipeline as SKL_Pipeline
from sklearn import svm, grid_search, datasets


from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext

from pdspark.converter import Converter
from pdspark.grid_search import GridSearchCV

sc = SparkContext('local[4]', "spark-sklearn tests")

class MLlibTestCase(unittest.TestCase):
    def setUp(self):
        self.sc = sc
        self.sql = SQLContext(sc)
        self.X = np.array([[1,2,3],
                           [-1,2,3], [1,-2,3], [1,2,-3],
                           [-1,-2,3], [1,-2,-3], [-1,2,-3],
                           [-1,-2,-3]])
        self.y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        data = [(float(self.y[i]), Vectors.dense(self.X[i])) for i in range(len(self.y))]
        self.df = self.sql.createDataFrame(data, ["label", "features"])

    @staticmethod
    def list2csr(x):
        """
        Convert list to a scipy.sparse.csr_matrix
        :param data: list
        :return:  csr_matrix with 1 row
        """
        return csr_matrix((np.array(x), np.array(range(0, len(x))), np.array([0, len(x)])))


class ConverterTests(MLlibTestCase):

    def setUp(self):
        super(ConverterTests, self).setUp()
        self.converter = Converter(self.sc)

    def _compare_GLMs(self, skl, spark):
        """ Compare weights, intercept of sklearn, Spark GLMs
        """
        skl_weights = Vectors.dense(skl.coef_.flatten())
        self.assertEqual(skl_weights, spark.weights)
        self.assertEqual(skl.intercept_, spark.intercept)

    def test_LogisticRegression_skl2spark(self):
        skl_lr = SKL_LogisticRegression().fit(self.X, self.y)
        lr = self.converter.toSpark(skl_lr)
        self.assertTrue(isinstance(lr, LogisticRegressionModel),
                        "Expected LogisticRegressionModel but found type %s" % type(lr))
        self._compare_GLMs(skl_lr, lr)

    def test_LinearRegression_skl2spark(self):
        skl_lr = SKL_LinearRegression().fit(self.X, self.y)
        lr = self.converter.toSpark(skl_lr)
        self.assertTrue(isinstance(lr, LinearRegressionModel),
                        "Expected LinearRegressionModel but found type %s" % type(lr))
        self._compare_GLMs(skl_lr, lr)

    def test_LogisticRegression_spark2skl(self):
        lr = LogisticRegression().fit(self.df)
        skl_lr = self.converter.toSKLearn(lr)
        self.assertTrue(isinstance(skl_lr, SKL_LogisticRegression),
                        "Expected sklearn LogisticRegression but found type %s" % type(skl_lr))
        self._compare_GLMs(skl_lr, lr)

    def test_LinearRegression_spark2skl(self):
        lr = LinearRegression().fit(self.df)
        skl_lr = self.converter.toSKLearn(lr)
        self.assertTrue(isinstance(skl_lr, SKL_LinearRegression),
                        "Expected sklearn LinearRegression but found type %s" % type(skl_lr))
        self._compare_GLMs(skl_lr, lr)

    def ztest_toPandas(self):
        data = [(Vectors.dense([0.1, 0.2]),),
                (Vectors.sparse(2, {0:0.3, 1:0.4}),),
                (Vectors.sparse(2, {0:0.5, 1:0.6}),)]
        df = self.sql.createDataFrame(data, ["features"])
        self.assertEqual(df.count(), 3)
        pd = self.converter.toPandas(df)
        self.assertEqual(len(pd), 3)
        self.assertTrue(isinstance(pd.features[0], csr_matrix),
                        "Expected pd.features[0] to be csr_matrix but found: %s" %
                        type(pd.features[0]))
        self.assertEqual(pd.features[0].shape[0], 3)
        self.assertEqual(pd.features[0].shape[1], 2)
        self.assertEqual(pd.features[0][0,0], 0.1)
        self.assertEqual(pd.features[0][0,1], 0.2)


class CSRVectorUDTTests(MLlibTestCase):

    def test_scipy_sparse(self):
        data = [(self.list2csr([0.1, 0.2]),)]
        df = self.sql.createDataFrame(data, ["features"])
        self.assertEqual(df.count(), 1)
        pd = df.toPandas()
        self.assertEqual(len(pd), 1)
        self.assertTrue(isinstance(pd.features[0], csr_matrix),
                        "Expected pd.features[0] to be csr_matrix but found: %s" %
                        type(pd.features[0]))
        self.assertEqual(pd.features[0].shape[0], 1)
        self.assertEqual(pd.features[0].shape[1], 2)
        self.assertEqual(pd.features.values[0][0,0], 0.1)
        self.assertEqual(pd.features.values[0][0,1], 0.2)

class CVTests2(MLlibTestCase):

    def setUp(self):
        super(CVTests2, self).setUp()
        self.converter = Converter(self.sc)

    def test_example(self):
        # The classic example from the sklearn documentation
        iris = datasets.load_iris()
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svr = svm.SVC()
        clf = grid_search.GridSearchCV(svr, parameters)
        clf.fit(iris.data, iris.target)

        clf2 = GridSearchCV(sc, svr, parameters)
        clf2.fit(iris.data, iris.target)

        b1 = clf.estimator
        b2 = clf2.estimator
        self.assertEqual(b1.get_params(), b2.get_params())


class CVTests(MLlibTestCase):

    def setUp(self):
        super(CVTests, self).setUp()
        self.converter = Converter(self.sc)

    def test_cv_linreg(self):
        pipeline = SKL_Pipeline([
            ('lasso', SKL_Lasso(max_iter=1))
        ])
        parameters = {
            'lasso__alpha': (0.001, 0.005, 0.01)
        }
        grid_search = GridSearchCV(self.sc, pipeline, parameters)
        X = scipy.sparse.vstack(map(lambda x: self.list2csr([x, x+1.0]), range(0, 100)))
        y = np.array(list(range(0, 100))).reshape((100,1))
        skl_gs = grid_search.fit(X, y)
        assert isinstance(skl_gs, SKL_GridSearchCV),\
            "GridSearchCV expected git to return a scikit-learn GridSearchCV instance," \
            " but found type %s" % type(skl_gs)
        assert len(skl_gs.grid_scores_) == len(parameters['lasso__alpha'])
        # TODO
        for gs in skl_gs.grid_scores_:
            pass # assert(gs.)

    def test_cv_pipeline(self):
        pipeline = SKL_Pipeline([
            ('vect', SKL_HashingVectorizer(n_features=20)),
            ('tfidf', SKL_TfidfTransformer(use_idf=False)),
            ('lasso', SKL_Lasso(max_iter=1))
        ])
        parameters = {
            'lasso__alpha': (0.001, 0.005, 0.01)
        }
        grid_search = GridSearchCV(self.sc, pipeline, parameters)
        data = [('hi there', 0.0),
                ('what is up', 1.0),
                ('huh', 1.0),
                ('now is the time', 5.0),
                ('for what', 0.0),
                ('the spark was there', 5.0),
                ('and so', 3.0),
                ('were many socks', 0.0),
                ('really', 1.0),
                ('too cool', 2.0)]
        df = self.sql.createDataFrame(data, ["review", "rating"]).toPandas()
        skl_gs = grid_search.fit(df.review.values, df.rating.values)
        assert isinstance(skl_gs, SKL_GridSearchCV), \
            "GridSearchCV expected git to return a scikit-learn GridSearchCV instance," \
            " but found type %s" % type(skl_gs)
        assert len(skl_gs.grid_scores_) == len(parameters['lasso__alpha'])
        # TODO
        for gs in skl_gs.grid_scores_:
            pass # assert(gs.)

    def test_cv_lasso_with_mllib_featurization(self):
        data = [('hi there', 0.0),
                ('what is up', 1.0),
                ('huh', 1.0),
                ('now is the time', 5.0),
                ('for what', 0.0),
                ('the spark was there', 5.0),
                ('and so', 3.0),
                ('were many socks', 0.0),
                ('really', 1.0),
                ('too cool', 2.0)]
        data = self.sql.createDataFrame(data, ["review", "rating"])

        # Feature extraction using MLlib
        tokenizer = Tokenizer(inputCol="review", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20000)
        pipeline = Pipeline(stages=[tokenizer, hashingTF])
        data = pipeline.fit(data).transform(data)

        df = self.converter.toPandas(data.select(data.features.alias("review"), "rating"))

        pipeline = SKL_Pipeline([
            ('lasso', SKL_Lasso(max_iter=1))
        ])
        parameters = {
            'lasso__alpha': (0.001, 0.005, 0.01)
        }

        grid_search = GridSearchCV(self.sc, pipeline, parameters)
        skl_gs = grid_search.fit(df.review.values, df.rating.values)

        assert isinstance(skl_gs, SKL_GridSearchCV), \
            "GridSearchCV expected git to return a scikit-learn GridSearchCV instance," \
            " but found type %s" % type(skl_gs)
        assert len(skl_gs.grid_scores_) == len(parameters['lasso__alpha'])
        # TODO
        for gs in skl_gs.grid_scores_:
            pass # assert(gs.)

    @unittest.skip("demo, disabled because it requires some specific files from Joseph")
    def test_demo(self):
        print "\n========================"
        print "DEMO PART 1"
        print "========================\n"
        trainFilepath = "/Users/josephkb/Desktop/demotest"
        data = self.sql.read.format("json").load(trainFilepath)
        df = data.toPandas()
        print "%d reviews" % len(df)

        reviews = df.review.values
        ratings = df.rating.values
        # Train scikit-learn model
        pipeline = SKL_Pipeline([
            ('vect', SKL_HashingVectorizer(n_features=100)),
            ('tfidf', SKL_TfidfTransformer(use_idf=False)),
            ('lasso', SKL_Lasso(max_iter=2)),
        ])
        parameters = {
            'lasso__alpha': (0.001, 0.005, 0.01)
        }

        nfolds = 3

        grid_search = SKL_GridSearchCV(pipeline, parameters, cv=nfolds)
        grid_search.fit(reviews, ratings)
        print("Best (training) R^2 score: %g" % grid_search.best_score_)
        pipeline = grid_search.best_estimator_

        r2 = pipeline.score(reviews, ratings)
        print "Training data R^2 score: %g" % r2

        test = self.sql.read.format("json").load(trainFilepath).toPandas()
        r2 = pipeline.score(test.review.values, test.rating.values)
        print "Test data R^2 score: %g" % r2

        predictions = pipeline.predict(test.review.values)
        pdPredictions = pandas.DataFrame(predictions)
        sparkPredictions = self.sql.createDataFrame(pdPredictions)
        print "sparkPredictions.count(): %d" % sparkPredictions.count()

        print "\n========================"
        print "DEMO PART 2"
        print "========================\n"

        grid_search = GridSearchCV(self.sc, pipeline, parameters, cv=nfolds)
        grid_search.fit(reviews, ratings)
        grid_search = grid_search.sklearn_model_
        print("Best (training) R^2 score: %g" % grid_search.best_score_)
        pipeline = grid_search.best_estimator_

        r2 = pipeline.score(reviews, ratings)
        print "Training data R^2 score: %g" % r2

        test = self.sql.read.format("json").load(trainFilepath).toPandas()
        r2 = pipeline.score(test.review.values, test.rating.values)
        print "Test data R^2 score: %g" % r2

        predictions = pipeline.predict(test.review.values)
        pdPredictions = pandas.DataFrame(predictions)
        sparkPredictions = self.sql.createDataFrame(pdPredictions)
        print "sparkPredictions.count(): %d" % sparkPredictions.count()

        print "\n========================"
        print "DEMO PART 3"
        print "========================\n"

        print "%d reviews" % data.count()

        # Define a pipeline combining text feature extractors
        tokenizer = Tokenizer(inputCol="review", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100)
        pipeline = Pipeline(stages=[tokenizer, hashingTF])
        data2 = pipeline.fit(data).transform(data)

        df = self.converter.toPandas(data2.select(data2.features.alias("review"), "rating"))

        reviews = self.converter.toScipy(df.review.values)
        ratings = df.rating.values

        pipeline = SKL_Pipeline([
            ('lasso', SKL_Lasso(max_iter=2)),
        ])

        grid_search = GridSearchCV(self.sc, pipeline, parameters, cv=nfolds)
        grid_search.fit(reviews, ratings)
        grid_search = grid_search.sklearn_model_
        print("Best (training) R^2 score: %g" % grid_search.best_score_)
        pipeline = grid_search.best_estimator_

        r2 = pipeline.score(reviews, ratings)
        print "Training data R^2 score: %g" % r2

        # Skip testing for this part of demo

        print "\n========================"
        print "DEMO PART 4"
        print "========================\n"

        # Define a pipeline combining a text feature extractor with a simple regressor
        tokenizer = Tokenizer(inputCol="review", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100)
        lasso = LinearRegression(labelCol="rating", elasticNetParam=1.0, maxIter=2)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lasso])

        paramGrid = ParamGridBuilder() \
            .addGrid(lasso.regParam, [0.001, 0.005, 0.01]) \
            .build()

        evaluator = RegressionEvaluator(labelCol="rating", metricName="r2")

        cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid)

        cvModel = cv.fit(data)

        test = self.sql.read.format("json").load(trainFilepath)
        r2 = evaluator.evaluate(cvModel.transform(test))
        print "Test data R^2 score: %g" % r2


if __name__ == "__main__":
    unittest.main()
    sc.stop()
