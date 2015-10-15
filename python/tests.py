
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
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.regression import LinearRegressionModel
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

class ConverterTests(MLlibTestCase):

    def setUp(self):
        super(ConverterTests, self).setUp()
        self.converter = Converter(self.sc)

    def test_LogisticRegression(self):
        skl_lr = SKL_LogisticRegression().fit(self.X, self.y)
        skl_weights = Vectors.dense(skl_lr.coef_.flatten())
        skl_intercept = skl_lr.intercept_
        lr = self.converter.toSpark(skl_lr)
        self.assertTrue(isinstance(lr, LogisticRegressionModel),
                        "Expected LogisticRegressionModel but found type %s" % type(lr))
        spark_weights = lr.weights
        spark_intercept = lr.intercept
        self.assertEqual(skl_weights, spark_weights)
        self.assertEqual(skl_intercept, spark_intercept)

    def test_LinearRegression(self):
        skl_lr = SKL_LinearRegression().fit(self.X, self.y)
        skl_weights = Vectors.dense(skl_lr.coef_.flatten())
        skl_intercept = skl_lr.intercept_
        lr = self.converter.toSpark(skl_lr)
        self.assertTrue(isinstance(lr, LinearRegressionModel),
                        "Expected LinearRegressionModel but found type %s" % type(lr))
        spark_weights = lr.weights
        spark_intercept = lr.intercept
        self.assertEqual(skl_weights, spark_weights)
        self.assertEqual(skl_intercept, spark_intercept)


if __name__ == "__main__":
    unittest.main()
    sc.stop()
