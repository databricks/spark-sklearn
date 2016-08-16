
import pandas as pd
import random
import unittest

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.tests import PythonOnlyPoint, PythonOnlyUDT, ExamplePoint, ExamplePointUDT

from spark_sklearn.test_utils import fixtureReuseSparkSession, assertPandasAlmostEqual, RandomTest
from spark_sklearn import gapply

def _assertPandasAlmostEqual(actual, expected):
    # Points are unhashable, so pandas' assert_frame_equal can't check they're the same by default.
    # We need to convert them to something that can be checked.
    def convert_points(pt):
        if any(isinstance(pt, t) for t in [PythonOnlyPoint, ExamplePoint]):
            return (pt.x, pt.y)
        return pt
    assertPandasAlmostEqual(actual, expected, convert=convert_points)

def _emptyFunc(key, vals):
    return pd.DataFrame.from_records([])

@fixtureReuseSparkSession
class GapplyTests(RandomTest):

    def test_gapply_empty(self):
        # Implicitly checks that pandas version is large enough (unit tests for the actual version
        # checking itself would require some serious mocking)
        longLongSchema = StructType().add("a", LongType()).add("b", LongType())
        emptyLongLongDF = self.spark.createDataFrame([], schema=longLongSchema)
        gd = emptyLongLongDF.groupBy("a")
        self.assertEqual(gapply(gd, _emptyFunc, longLongSchema, "b").collect(), [])

    def test_gapply_empty_schema(self):
        longLongSchema = StructType().add("a", LongType()).add("b", LongType())
        emptyLongLongDF = self.spark.createDataFrame([(1, 2)], schema=longLongSchema)
        gd = emptyLongLongDF.groupBy("a")
        self.assertEqual(gapply(gd, _emptyFunc, StructType(), "b").collect(), [])

    def test_gapply_raises_if_bad_schema(self):
        longLongSchema = StructType().add("a", LongType()).add("b", LongType())
        emptyLongLongDF = self.spark.createDataFrame([], schema=longLongSchema)
        gd = emptyLongLongDF.groupBy("a")
        self.assertRaises(ValueError, gapply, gd, _emptyFunc, LongType(), "b")

    def test_gapply_raises_if_bad_cols(self):
        longLongLongSchema = StructType() \
                             .add("a", LongType()).add("b", LongType()).add("c", LongType())
        emptyLongLongLongDF = self.spark.createDataFrame([], schema=longLongLongSchema)
        gd = emptyLongLongLongDF.groupBy("a")
        self.assertRaises(ValueError, gapply, gd, _emptyFunc, longLongLongSchema, "b", "*")
        self.assertRaises(ValueError, gapply, gd, _emptyFunc, longLongLongSchema, "*")
        self.assertRaises(ValueError, gapply, gd, _emptyFunc, longLongLongSchema, "b", "b")
        self.assertRaises(ValueError, gapply, gd, _emptyFunc, longLongLongSchema, "b", "c", "b")

    NROWS = 15
    NKEYS = 5
    NVALS = 100

    def checkGapplyEquivalentToPandas(self, pandasAggFunction, dataType, dataGen):
        schema = StructType().add("val", dataType)
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(GapplyTests.NKEYS) for _ in range(GapplyTests.NROWS)],
            "val": [dataGen() for _ in range(GapplyTests.NROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            return pd.DataFrame.from_records([(pandasAggFunction(vals["val"]),)])
        expected = pandasDF.groupby("key", as_index=False).agg({"val": pandasAggFunction})
        actual = gapply(gd, func, schema, "val").toPandas()
        _assertPandasAlmostEqual(actual, expected)

    def test_gapply_primitive_val(self):
        pandasAggFunction = lambda series: series.sum()
        dataType = LongType()
        dataGen = lambda: random.randrange(GapplyTests.NVALS)
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    def test_gapply_struct_val(self):
        def pandasAggFunction(series):
            x = int(series.apply(sum).sum()) # nested dtypes aren't converted, it's on the user
            return (x, x)
        dataType = StructType().add("a", LongType()).add("b", LongType())
        dataGen = lambda: (random.randrange(GapplyTests.NVALS), random.randrange(GapplyTests.NVALS))
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    @unittest.skip("""
    python only UDTs can't be nested in arraytypes for now, see SPARK-15989
    this is only available starting in Spark 2.0.1
    """)
    def test_gapply_python_only_udt_val(self):
        def pandasAggFunction(series):
            x = float(series.apply(lambda pt: int(pt.x) + int(pt.y)).sum())
            return PythonOnlyPoint(x, x) # still deterministic, can have exact equivalence test
        dataType = PythonOnlyUDT()
        dataGen = lambda: PythonOnlyPoint(
            float(random.randrange(GapplyTests.NVALS)),
            float(random.randrange(GapplyTests.NVALS)))
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    def test_gapply_universal_udt_val(self):
        def pandasAggFunction(series):
            x = float(series.apply(lambda pt: int(pt.x) + int(pt.y)).sum())
            return ExamplePoint(x, x) # still deterministic, can have exact equivalence test
        dataType = ExamplePointUDT()
        dataGen = lambda: ExamplePoint(
            float(random.randrange(GapplyTests.NVALS)),
            float(random.randrange(GapplyTests.NVALS)))
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    def test_gapply_double_key(self):
        schema = StructType().add("val", LongType())
        randKeys = [random.randrange(GapplyTests.NKEYS) for _ in range(GapplyTests.NROWS)]
        pandasDF = pd.DataFrame.from_dict({
            "key1": randKeys,
            "key2": [GapplyTests.NKEYS + x for x in randKeys],
            "val": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key2", "key1")
        def func(keys, vals):
            assert keys[0] == keys[1] + GapplyTests.NKEYS
            return pd.DataFrame.from_records([(vals["val"].sum(),)])
        expected = pandasDF.groupby(["key2", "key1"], as_index=False).agg({"val": "sum"})
        actual = gapply(gd, func, schema, "val").toPandas()
        _assertPandasAlmostEqual(actual, expected)

    def test_gapply_name_change(self):
        schema = StructType().add("VAL", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(GapplyTests.NKEYS) for _ in range(GapplyTests.NROWS)],
            "val": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            return pd.DataFrame.from_records([(vals["val"].sum(),)])
        expected = pandasDF.groupby("key", as_index=False).agg({"val": "sum"})
        expected = expected.rename(columns={"val": "VAL"})
        actual = gapply(gd, func, schema, "val").toPandas()
        _assertPandasAlmostEqual(actual, expected)

    def test_gapply_one_col(self):
        schema = StructType().add("val2", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(GapplyTests.NKEYS) for _ in range(GapplyTests.NROWS)],
            "val1": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)],
            "val2": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            assert vals.columns.tolist() == ["val2"], vals.columns
            return pd.DataFrame.from_records([(vals["val2"].sum(),)])
        expected = pandasDF.groupby("key", as_index=False).agg({"val2": "sum"})
        actual = gapply(gd, func, schema, "val2").toPandas()
        _assertPandasAlmostEqual(actual, expected)

    def test_gapply_all_cols(self):
        schema = StructType().add("val2", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(GapplyTests.NKEYS) for _ in range(GapplyTests.NROWS)],
            "val1": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)],
            "val2": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)]})
        df = self.spark.createDataFrame(pandasDF)
        gd = df.groupBy("key")
        def func(key, vals):
            assert vals.columns.tolist() == ["val1", "val2"], vals.columns
            return pd.DataFrame.from_records([(vals["val2"].sum(),)])
        expected = pandasDF.groupby("key", as_index=False).agg({"val2": "sum"})
        actual = gapply(gd, func, schema).toPandas()
        _assertPandasAlmostEqual(actual, expected)
        def func(key, vals):
            assert vals.columns.tolist() == ["val2", "val1"], vals.columns
            return pd.DataFrame.from_records([(vals["val2"].sum(),)])
        gd = df.select("val2", "key", "val1").groupBy("key")
        actual = gapply(gd, func, schema).toPandas()
        _assertPandasAlmostEqual(actual, expected)

@fixtureReuseSparkSession
class GapplyConfTests(RandomTest):
    @classmethod
    def setUpClass(cls):
        super(GapplyConfTests, cls).setUpClass()
        cls.spark = SparkSession.builder \
                                .config("spark.sql.retainGroupColumns", "false") \
                                .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        super(GapplyConfTests, cls).tearDownClass()
        # Creating a new SparkSession here seems confusing, but it is necessary because
        # the config is (for some stupid reason...) cached, which would make it get in
        # the way of other tests that expect a default configuration.
        cls.spark = SparkSession.builder \
                                .config("spark.sql.retainGroupColumns", "true") \
                                .getOrCreate()

    def test_gapply_no_keys(self):
        schema = StructType().add("val", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(GapplyTests.NKEYS) for _ in range(GapplyTests.NROWS)],
            "val": [random.randrange(GapplyTests.NVALS) for _ in range(GapplyTests.NROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(_, vals):
            return pd.DataFrame.from_records([(vals["val"].sum(),)])
        expected = pandasDF.groupby("key", as_index=False).agg({"val": "sum"})[["val"]]
        actual = gapply(gd, func, schema, "val").toPandas()
        _assertPandasAlmostEqual(actual, expected)
