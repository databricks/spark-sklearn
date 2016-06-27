"""
Some test utilities to create the spark context.
"""
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
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix
import time

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

from spark_sklearn.util import createLocalSparkSession

# Used as decorator to wrap around a class deriving from unittest.TestCase. Wraps current
# unittest methods setUpClass() and tearDownClass(), invoked by the nosetest command before
# and after unit tests are run. This enables us to create one PySpark SparkSession per
# test fixture. The session can be referred to with self.spark or ClassName.spark.
#
# The SparkSession is set up before invoking the class' own set up and torn down after the
# class' tear down, so you may safely refer to it in those methods.
def fixtureReuseSparkSession(cls):
    setup = getattr(cls, 'setUpClass', None)
    teardown = getattr(cls, 'tearDownClass', None)
    def setUpClass(cls):
        cls.spark = createLocalSparkSession("Unit Tests")
        if setup:
            setup()
    def tearDownClass(cls):
        if teardown:
            teardown()
        if cls.spark:
            cls.spark.stop()
            # Next session will attempt to reuse the previous stopped
            # SparkContext if it's not cleared.
            SparkSession._instantiatedContext = None
        cls.spark = None

    cls.setUpClass = classmethod(setUpClass)
    cls.tearDownClass = classmethod(tearDownClass)
    return cls

class MLlibTestCase(unittest.TestCase):
    def setUp(self):
        super(MLlibTestCase, self).setUp()
        self.sc = self.spark.sparkContext
        self.sql = self.spark
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

# Asserts that two Pandas dataframes are equal, with only 5 digits of precision for
# floats.
#
# If convert is not None, then applies convert to each item in both dataframes first.
#
# Sorts rows in dataframes by sortby. If sortby is None then all columns are used.
def assertPandasAlmostEqual(actual, expected, convert=None, sortby=None):
    def normalize(pdDF):
        converted = pdDF.apply(lambda col: col.apply(convert if convert else lambda x: x))
        ordered = converted.sort_values(sortby if sortby else pdDF.columns.tolist())
        # We need to drop the index after sorting because pandas remembers the pre-sort
        # permutation in the old index. This would trigger a failure if we were to compare
        # differently-ordered dataframes, even if they had the same sorted content.
        unindexed = ordered.reset_index(drop=True)
        return unindexed
    actual = normalize(actual)
    expected = normalize(expected)
    pd.util.testing.assert_almost_equal(actual, expected)

# This unittest.TestCase subclass sets the random seed to be based on the time
# that the test is run.
#
# If there is a SEED variable in the enviornment, then this is used as the seed.
# Sets both random and numpy.random.
#
# Prints the seed to stdout before running each test case.
class RandomTest(unittest.TestCase):
    def setUp(self):
        seed = os.getenv("SEED")
        seed = np.uint32(seed if seed else time.time())

        print('Random test using SEED={}'.format(seed))

        random.seed(seed)
        np.random.seed(seed)
