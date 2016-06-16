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
from scipy.sparse import csr_matrix

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors

# Used as deocrator to have one PySpark SparkSession per fixture.
def fixtureReuseSparkSession(cls):
    setup = getattr(cls, 'setUpClass', None)
    teardown = getattr(cls, 'tearDownClass', None)
    def setUpClass(cls):
        if setup: setup()
        cls.spark = SparkSession.builder.master("local").appName("Unit Tests").getOrCreate()
    def tearDownClass(cls):
        if cls.spark:
            cls.spark.stop()
            SparkSession._instantiatedContext = None
        cls.spark = None
        if teardown: teardown()

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
