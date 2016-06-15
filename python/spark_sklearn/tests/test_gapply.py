
from itertools import chain
import pandas as pd
import random
from pyspark.tests import ReusedPySparkTestCase
from pyspark.ml.param.shared import HasSeed


class GapplyTests(ReusedPySparkTestCase, HasSeed):
    
    @classmethod
    def setUpClass(cls):
        ReusedPySparkTestCase.setUpClass()
        cls.spark = SparkSession(cls.sc)
        random.seed(1000) # random values for convenience, seed for repeatability
        
    @classmethod
    def tearDownClass(cls):
        ReusedPySparkTestCase.tearDownClass()
        cls.spark.stop()
    
    def emptyFunc(key, vals): return pd.DataFrame.from_records([])

    def test_gapply_empty(self):
        # Implicitly checks that pandas version is large enough (unit tests for the actual version
        # checking itself would require some serious mocking)
        longLongSchema = StructType().add("a", LongType()).add("b", LongType())
        emptyLongLongDF = self.spark.createDataFrame([], schema=longLongSchema)
        gd = emptyLongLongDF.groupBy("a")
        self.assertEqual(gd.gapply(emptyFunc, longLongSchema, "b").collect(), [])

    def test_gapply_raises_if_bad_cols(self):
        longLongLongSchema = StructType().add("a", LongType()).add("b", LongType()).add("c", LongType())
        emptyLongLongLongDF = self.spark.createDataFrame([], schema=longLongLongSchema)
        gd = emptyLongLongLongDF.groupBy("a")
        self.assertRaises(ValueError, gd.gapply(emptyFunc, longLongLongSchema, "b", "*"))
        self.assertRaises(ValueError, gd.gapply(emptyFunc, longLongLongSchema, "*"))
        self.assertRaises(ValueError, gd.gapply(emptyFunc, longLongLongSchema, "b", "b"))
        self.assertRaises(ValueError, gd.gapply(emptyFunc, longLongLongSchema, "b", "c", "b"))

    NUM_TEST_ROWS = 15
    NUM_TEST_KEYS = 5
    NUM_TEST_VALS = 100

    def checkGapplyEquivalentToPandas(self, pandasAggFunction, dataType, dataGen):
        schema = StructType().add("key", LongType()).add("val", dataType)
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(NUM_TEST_KEYS) for _ in range(NUM_TEST_ROWS)],
            "val": [dataGen() for _ in range(NUM_TEST_ROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            return pd.DataFrame.from_records([(key, pandasAggFunction(vals["val"]))])
        expected = pandasDF.groupBy("key").agg({"val": pandasAggFunction})
        actual = gd.gapply(func, schema, "val").toPandas()
        pd.utils.assert_frame_equal(actual, expected)

    def test_gapply_primitive_val(self):
        pandasAggFunction = lambda series: series.sum()
        dataType = LongType()
        dataGen = lambda: random.randrange(NUM_TEST_VALS)
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    def test_gapply_struct_val(self):
        def pandasAggFunction(series):
            sum = series.apply(sum).sum()
            return (sum, sum)
        dataType = StructType().add("a", LongType()).add("b", LongType())
        dataGen = lambda: (random.randrange(NUM_TEST_VALS), random.randrange(NUM_TEST_VALS))
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    def test_gapply_udt_val(self):
        def pandasAggFunction(series):
            sum = series.apply(lambda pt: int(pt.x) + int(pt.y))
            return PythonOnlyPoint(sum, sum) # Still deterministic, can have exact equivalence test
        dataType = PythonOnlyUDT()
        dataGen = lambda: PythonOnlyPoint(random.randrange(NUM_TEST_VALS), random.randrange(NUM_TEST_VALS))
        self.checkGapplyEquivalentToPandas(pandasAggFunction, dataType, dataGen)

    def test_gapply_double_key(self):
        schema = StructType().add("key1", LongType()).add("key2", LongType()).add("val", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key1": [random.randrange(NUM_TEST_KEYS // 2) for _ in range(NUM_TEST_ROWS)],
            "key2": [random.randrange(NUM_TEST_KEYS // 2) for _ in range(NUM_TEST_ROWS)],
            "val": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key1", "key2")
        def func(key1, key2, vals):
            return pd.DataFrame.from_records([(key1, key2, vals["val"].sum())])
        expected = pandasDF.groupBy("key").agg({"val": "sum"})
        actual = gd.gapply(func, schema, "val").toPandas()
        pd.utils.assert_frame_equal(actual, expected)

    def test_gapply_name_change(self):
        schema = StructType().add("KEY", LongType()).add("VAL", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(NUM_TEST_KEYS) for _ in range(NUM_TEST_ROWS)],
            "val": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            return pd.DataFrame.from_records([(key, vals["val"].sum())])
        expected = pandasDF.groupBy("key").agg({"val": "sum"}).rename(columns={"key": "KEY", "val": "VAL"})
        actual = gd.gapply(func, schema, "val").toPandas()
        pd.utils.assert_frame_equal(actual, expected)

    def test_gapply_one_col(self):
        schema = StructType().add("key", LongType()).add("val2", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(NUM_TEST_KEYS) for _ in range(NUM_TEST_ROWS)],
            "val1": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)],
            "val2": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            assert vals.columns == ["val2"]
            return pd.DataFrame.from_records([(key, vals["val2"].sum())])
        expected = pandasDF.groupBy("key").agg({"val2": "sum"})
        actual = gd.gapply(func, schema, "val2").toPandas()
        pd.utils.assert_frame_equal(actual, expected)

    def test_gapply_all_cols(self):
        schema = StructType().add("key", LongType()).add("val2", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(NUM_TEST_KEYS) for _ in range(NUM_TEST_ROWS)],
            "val1": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)],
            "val2": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)]})
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        def func(key, vals):
            assert vals.columns == ["val1", "val2"]
            return pd.DataFrame.from_records([(key, vals["val2"].sum())])
        expected = pandasDF.groupBy("key").agg({"val2": "sum"})
        actual = gd.gapply(func, schema).toPandas()
        pd.utils.assert_frame_equal(actual, expected)
        def func(key, vals):
            assert vals.columns == ["val2", "val1"]
            return pd.DataFrame.from_records([(key, vals["val2"].sum())])
        expected = pandasDF.select("val2", "key", "val1").gapply(func, schema)
        pd.utils.assert_frame_equal(actual, expected)

    def test_gapply_no_keys(self):
        schema = StructType().add("key", LongType()).add("val", LongType())
        pandasDF = pd.DataFrame.from_dict({
            "key": [random.randrange(NUM_TEST_KEYS) for _ in range(NUM_TEST_ROWS)],
            "val": [random.randrange(NUM_TEST_VALS) for _ in range(NUM_TEST_ROWS)]})
        oldConf = self.spark.getConf("spark.sql.retainGroupColumns")
        self.spark.setConf("spark.sql.retainGroupColumns", "false")
        gd = self.spark.createDataFrame(pandasDF).groupBy("key")
        self.spark.setConf("spark.sql.retainGroupColumns", oldConf)
        def func(vals):
            return pd.DataFrame.from_records([(key, vals["val"].sum())])
        expected = pandasDF.groupBy("key").agg({"val": "sum"})
        actual = gd.gapply(func, schema, "val").toPandas()[["val"]]
        pd.utils.assert_frame_equal(actual, expected)
