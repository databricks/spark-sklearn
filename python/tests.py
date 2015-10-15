
import sys

if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest

import numpy

from sklearn.linear_model import LogisticRegression as SKL_LogisticRegression
from sklearn.linear_model import LinearRegression as SKL_LinearRegression

from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext

from converter import Converter

sc = SparkContext('local[4]', "spark-sklearn tests")

class MLlibTestCase(unittest.TestCase):
    def setUp(self):
        self.sc = sc
        self.sql = SQLContext(sc)
        self.X = numpy.array([[1,2,3],
                              [-1,2,3], [1,-2,3], [1,2,-3],
                              [-1,-2,3], [1,-2,-3], [-1,2,-3],
                              [-1,-2,-3]])
        self.y = numpy.array([1, 0, 1, 1, 0, 1, 0, 0])
        data = [(float(self.y[i]), Vectors.dense(self.X[i])) for i in range(len(self.y))]
        self.df = self.sql.createDataFrame(data, ["label", "features"])

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


if __name__ == "__main__":
    unittest.main()
    sc.stop()
