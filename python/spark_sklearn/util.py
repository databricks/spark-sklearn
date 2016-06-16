
import uuid

from distutils.version import LooseVersion
from py4j import java_gateway
from pyspark import SparkContext, since
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import *
from itertools import chain


# WARNING: These are private Spark APIs.
from pyspark.ml.common import _py2java, _java2py

def _jvm():
    """
    Returns the JVM view associated with SparkContext. Must be called
    after SparkContext is initialized.
    """
    jvm = SparkContext._jvm
    if jvm:
        return jvm
    else:
        raise AttributeError("Cannot load _jvm from SparkContext. Is SparkContext initialized?")

def _new_java_obj(sc, java_class, *args):
    """
    Construct a new Java object.
    """
    java_obj = _jvm()
    for name in java_class.split("."):
        java_obj = getattr(java_obj, name)
    java_args = [_py2java(sc, arg) for arg in args]
    return java_obj(*java_args)

def _call_java(sc, java_obj, name, *args):
    """
    Method copied from pyspark.ml.wrapper.  Uses private Spark APIs.
    """
    m = getattr(java_obj, name)
    java_args = [_py2java(sc, arg) for arg in args]
    return _java2py(sc, m(*java_args))

def _randomUID(cls):
    """
    Generate a unique id for the object. The default implementation
    concatenates the class name, "_", and 12 random hex chars.
    """
    return cls.__name__ + "_" + uuid.uuid4().hex[12:]

@since(2.0)
def gapply(grouped_data, func, schema, *cols):
    """Applies the function `func` to the grouped data; in particular, by default this calls
    `func(key1, key2, ..., keyn, values)` where the number and order of the key arguments is
    determined by columns on which this instance's parent :class:`DataFrame` was grouped and
    `values` is a `pandas.DataFrame` of columns selected by `cols`, in that order.

    `func` is expected to return a `pandas.DataFrame` of the specified schema `schema`,
    which should be of type :class:`StructType` (output columns are of this name and order).

    If `spark.conf.get("spark.sql.retainGroupColumns")` is not `u'true'`, then `func` is
    called without any keys.

    If no `cols` are specified, then all grouped columns will be offered, in the order of the 
    columns in the original dataframe. In either case, the Pandas columns will be named 
    according to the DataFrame column names.

    The order of the rows passed in as Pandas rows is not guaranteed to be stable relative to 
    the original row order.

    .. :note: Users must ensure that the grouped values for every group must fit entirely in
    memory.
    .. :note: This method is only available if Pandas is installed.

    :param func: a two argument closure, which may be either a lambda or named function
    :param schema: the return schema for `func`
    :param cols: list of column names (string only)

    :raise ValueError: if `*` is in `cols`
    :raise ValueError: if `cols` contains duplicates
    :raise ValueError: if `schema` is not a :class:`StructType`
    :raise ImportError: if `pandas` module is not installed
    :raise ImportError: if `pandas` version is too old (less than 0.7.1)

    :return: a new :class:`DataFrame` of schema `schema` formed from concatenation of the
    returned data frames.

    >>> import pandas as pd
    >>> df = spark \
    ...          .createDataFrame([Row(course="dotNET", year=2012, earnings=10000),
    ...                            Row(course="Java",   year=2012, earnings=20000),
    ...                            Row(course="dotNET", year=2012, earnings=5000),
    ...                            Row(course="dotNET", year=2013, earnings=48000),
    ...                            Row(course="Java",   year=2013, earnings=30000)]) \
    ...          .select("course", "year", "earnings")
    DataFrame[course: string, year: bigint, earnings: bigint]
    >>> def yearlyMedian(key, vals):
    ...     all_years = set(vals['year'])
    ...     # Note that interpolation is performed, so we need to cast back to long.
    ...     yearly_median = [(key, year, long(vals['earnings'][vals['year'] == year].median()))
    ...                      for year in all_years]
    ...     return pd.DataFrame.from_records(yearly_median)
    >>> gapply(df.groupBy("course"), yearlyMedian, df.schema).show()
    +------+----+--------+
    |course|year|earnings|
    +------+----+--------+
    |dotNET|2012|    7500|
    |dotNET|2013|   48000|
    |  Java|2012|   20000|
    |  Java|2013|   30000|
    +------+----+--------+
    >>> def twoKeyYearlyMedian(course, year, vals):
    ...     return pd.DataFrame.from_records([(course, year, long(vals["earnings"].median()))])
    >>> gapply(df.groupBy("course", "year"), twoKeyYearlyMedian, df.schema, "earnings").show()
    +------+----+--------+
    |course|year|earnings|
    +------+----+--------+
    |dotNET|2012|    7500|
    |dotNET|2013|   48000|
    |  Java|2012|   20000|
    |  Java|2013|   30000|
    +------+----+--------+
    """
    import pandas
    minPandasVersion = '0.7.1'
    if LooseVersion(pandas.__version__) < LooseVersion(minPandasVersion):
        raise ImportError('Pandas installed but version is {}, {} required'
                          .format(pandas.__version__, minPandasVersion))

    # Do a null aggregation to retrieve the keys first (should be no computation)
    key_cols = grouped_data.agg({}).columns

    if not cols:
        # Extract the full column list with the parent df
        javaDFName = "org$apache$spark$sql$RelationalGroupedDataset$$df"
        parentDF = java_gateway.get_field(grouped_data._jdf, javaDFName)
        all_cols = DataFrame(parentDF, None).columns
        key_cols_set = set(key_cols)
        cols = [col for col in all_cols if col not in key_cols_set]

    if "*" in cols:
        raise ValueError("cols expected to contain only singular columns")

    if len(set(cols)) < len(cols):
        raise ValueError("cols expected not to contain duplicate columns")

    if not isinstance(schema, StructType):
        raise ValueError("output schema should be a StructType")

    inputAggDF = grouped_data.agg({col: 'collect_list' for col in cols})
    # Recover canonical order (aggregation may change column order)
    cannonicalOrder = chain(key_cols, (inputAggDF['collect_list(' + col + ')'] for col in cols))
    inputAggDF = inputAggDF.select(*cannonicalOrder)

    def pandasWrappedFunc(*args):
        nvals = len(cols)
        keys, collectedCols = args[:-nvals], args[-nvals:]
        valuesDF = pandas.DataFrame.from_dict(
            {colName: colList for colName, colList in zip(cols, collectedCols)})
        valuesDF = valuesDF[list(cols)] # reorder to cannonical
        outputDF = func(*chain(keys, [valuesDF]))
        # To recover native python types for serialization, we need
        # to convert the pandas dataframe to a numpy array, then to a
        # native list (can't go straight to native, since pandas will
        # attempt to perserve the numpy type).
        return outputDF.values.tolist()

    outputAggSchema = ArrayType(schema, containsNull=False)
    pandasUDF = udf(pandasWrappedFunc, outputAggSchema)
    outputAggDF = inputAggDF.select(pandasUDF(*inputAggDF))

    explodedDF = outputAggDF.select(explode(*outputAggDF).alias("gapply"))
    # automatically retrieves nested schema column names
    return explodedDF.select("gapply.*")
