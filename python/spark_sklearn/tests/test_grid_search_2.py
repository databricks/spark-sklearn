import numpy as np
import scipy
from sklearn.linear_model import Lasso as SKL_Lasso
from sklearn.feature_extraction.text import HashingVectorizer as SKL_HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as SKL_TfidfTransformer
from sklearn.pipeline import Pipeline as SKL_Pipeline
from sklearn import svm, grid_search, datasets
import sys
if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest


from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer

from spark_sklearn.converter import Converter
from spark_sklearn.grid_search import GridSearchCV

from spark_sklearn.test_utils import MLlibTestCase, fixtureReuseSparkSession

@fixtureReuseSparkSession
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

        clf2 = GridSearchCV(self.sc, svr, parameters)
        clf2.fit(iris.data, iris.target)

        b1 = clf.estimator
        b2 = clf2.estimator
        self.assertEqual(b1.get_params(), b2.get_params())

@fixtureReuseSparkSession
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
        assert len(skl_gs.grid_scores_) == len(parameters['lasso__alpha'])
        # TODO
        for gs in skl_gs.grid_scores_:
            pass # assert(gs.)

    @unittest.skip("disable this test until we have numpy <-> dataframe conversion")
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

        assert len(skl_gs.grid_scores_) == len(parameters['lasso__alpha'])
        # TODO
        for gs in skl_gs.grid_scores_:
            pass
