
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression as SKL_LogisticRegression
from sklearn.linear_model import LinearRegression as SKL_LinearRegression
import unittest

from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from spark_sklearn.test_utils import MLlibTestCase, fixtureReuseSparkSession
from spark_sklearn import Converter

@fixtureReuseSparkSession
class ConverterTests(MLlibTestCase):

    def setUp(self):
        super(ConverterTests, self).setUp()
        self.converter = Converter(self.sc)

    def _compare_GLMs(self, skl, spark):
        """ Compare weights, intercept of sklearn, Spark GLMs
        """
        skl_weights = Vectors.dense(skl.coef_.flatten())
        self.assertEqual(skl_weights, spark.coefficients)
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

@fixtureReuseSparkSession
class CSRVectorUDTTests(MLlibTestCase):

    @unittest.skip("CSR Matrix support not present for Spark 2.0 - see issue #24")
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
