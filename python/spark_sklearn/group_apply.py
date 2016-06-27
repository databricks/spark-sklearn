"""
Method for applying arbitrary UDFs over grouped data.
"""

from distutils.version import LooseVersion
from itertools import chain
from py4j import java_gateway

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import *
from pyspark.sql.types import Row

def gapply(grouped_data, func, schema, *cols):
    """Applies the function ``func`` to data grouped by key. In particular, given a dataframe
    grouped by some set of key columns key1, key2, ..., keyn, this method groups all the values
    for each row with the same key columns into a single Pandas dataframe and by default invokes
    ``func((key1, key2, ..., keyn), values)`` where the number and order of the key arguments is
    determined by columns on which this instance's parent :class:`DataFrame` was grouped and
    ``values`` is a ``pandas.DataFrame`` of columns selected by ``cols``, in that order.

    If there is only one key then the key tuple is automatically unpacked, with
    ``func(key, values)`` called.

    ``func`` is expected to return a ``pandas.DataFrame`` of the specified schema ``schema``,
    which should be of type :class:`StructType` (output columns are of this name and order).

    If ``spark.conf.get("spark.sql.retainGroupColumns")`` is not ``u'true'``, then ``func`` is
    called with an empty key tuple (note it is set to ``u'true'`` by default).

    If no ``cols`` are specified, then all grouped columns will be offered, in the order of the
    columns in the original dataframe. In either case, the Pandas columns will be named
    according to the DataFrame column names.

    The order of the rows passed in as Pandas rows is not guaranteed to be stable relative to
    the original row order.

    :note: Users must ensure that the grouped values for every group must fit entirely in memory.
    :note: This method is only available if Pandas is installed.

    :param func: a two argument function, which may be either a lambda or named function
    :param schema: the return schema for ``func``, a :class:`StructType`
    :param cols: list of column names (string only)

    :raise ValueError: if ``"*"`` is in ``cols``
    :raise ValueError: if ``cols`` contains duplicates
    :raise ValueError: if ``schema`` is not a :class:`StructType`
    :raise ImportError: if ``pandas`` module is not installed
    :raise ImportError: if ``pandas`` version is too old (less than 0.7.1)

    :return: the new :class:`DataFrame` with the original key columns replicated for each returned
             value in each group's resulting pandas dataframe, the schema being the original key
             schema prepended to ``schema``, where all the resulting groups' rows are concatenated.
             Of course, if retaining group columns is disabled, then the output will exactly match
             ``schema`` since no keys can be prepended.

    >>> import pandas as pd
    >>> from pyspark.sql import SparkSession
    >>> from spark_sklearn.group_apply import gapply
    >>> from spark_sklearn.util import createLocalSparkSession
    >>> spark = createLocalSparkSession()
    >>> df = (spark
    ...     .createDataFrame([Row(course="dotNET", year=2012, earnings=10000),
    ...                       Row(course="Java",   year=2012, earnings=20000),
    ...                       Row(course="dotNET", year=2012, earnings=5000),
    ...                       Row(course="dotNET", year=2013, earnings=48000),
    ...                       Row(course="Java",   year=2013, earnings=30000)])
    ...     .select("course", "year", "earnings"))
    >>> def yearlyMedian(_, vals):
    ...     all_years = set(vals['year'])
    ...     # Note that interpolation is performed, so we need to cast back to int.
    ...     yearly_median = [(year, int(vals['earnings'][vals['year'] == year].median()))
    ...                      for year in all_years]
    ...     return pd.DataFrame.from_records(yearly_median)
    >>> newSchema = StructType().add("year", LongType()).add("median_earnings", LongType())
    >>> gapply(df.groupBy("course"), yearlyMedian, newSchema).orderBy("median_earnings").show()
    +------+----+---------------+
    |course|year|median_earnings|
    +------+----+---------------+
    |dotNET|2012|           7500|
    |  Java|2012|          20000|
    |  Java|2013|          30000|
    |dotNET|2013|          48000|
    +------+----+---------------+
    <BLANKLINE>
    >>> def twoKeyYearlyMedian(_, vals):
    ...     return pd.DataFrame.from_records([(int(vals["earnings"].median()),)])
    >>> newSchema = StructType([df.schema["earnings"]])
    >>> gapply(df.groupBy("course", "year"), twoKeyYearlyMedian, newSchema, "earnings").orderBy(
    ...     "earnings").show()
    +------+----+--------+
    |course|year|earnings|
    +------+----+--------+
    |dotNET|2012|    7500|
    |  Java|2012|   20000|
    |  Java|2013|   30000|
    |dotNET|2013|   48000|
    +------+----+--------+
    <BLANKLINE>
    >>> spark.stop(); SparkSession._instantiatedContext = None
    """
    import pandas as pd
    minPandasVersion = '0.7.1'
    if LooseVersion(pd.__version__) < LooseVersion(minPandasVersion):
        raise ImportError('Pandas installed but version is {}, {} required'
                          .format(pd.__version__, minPandasVersion))

    # Do a null aggregation to retrieve the keys first (should be no computation)
    # Also consistent with spark.sql.retainGroupColumns
    keySchema = grouped_data.agg({}).schema
    keyCols = grouped_data.agg({}).columns

    if not cols:
        # Extract the full column list with the parent df
        javaDFName = "org$apache$spark$sql$RelationalGroupedDataset$$df"
        parentDF = java_gateway.get_field(grouped_data._jgd, javaDFName)
        allCols = DataFrame(parentDF, None).columns
        keyColsSet = set(keyCols)
        cols = [col for col in allCols if col not in keyColsSet]

    if "*" in cols:
        raise ValueError("cols expected to contain only singular columns")

    if len(set(cols)) < len(cols):
        raise ValueError("cols expected not to contain duplicate columns")

    if not isinstance(schema, StructType):
        raise ValueError("output schema should be a StructType")

    inputAggDF = grouped_data.agg({col: 'collect_list' for col in cols})
    # Recover canonical order (aggregation may change column order)
    cannonicalOrder = chain(keyCols, [inputAggDF['collect_list(' + col + ')'] for col in cols])
    inputAggDF = inputAggDF.select(*cannonicalOrder)

    # Wraps the user-proveded function with another python function, which prepares the
    # input in the form specified by the documentation. Then, once the function completes,
    # this wrapper prepends the keys to the output values and converts from pandas.
    def pandasWrappedFunc(*args):
        nvals = len(cols)
        keys, collectedCols = args[:-nvals], args[-nvals:]
        paramKeys = tuple(keys)
        if len(paramKeys) == 1:
            paramKeys = paramKeys[0]
        valuesDF = pd.DataFrame.from_dict(dict(zip(cols, collectedCols)))
        valuesDF = valuesDF[list(cols)] # reorder to canonical
        outputDF = func(paramKeys, valuesDF)
        valCols = outputDF.columns.tolist()
        for key, keyName in zip(keys, keyCols):
            outputDF[keyName] = key
        outputDF = outputDF[keyCols + valCols] # reorder to canonical
        # To recover native python types for serialization, we need
        # to convert the pandas dataframe to a numpy array, then to a
        # native list (can't go straight to native, since pandas will
        # attempt to perserve the numpy type).
        return outputDF.values.tolist()

    keyPrependedSchema = StructType(list(chain(keySchema, schema)))
    outputAggSchema = ArrayType(keyPrependedSchema, containsNull=False)
    pandasUDF = udf(pandasWrappedFunc, outputAggSchema)
    outputAggDF = inputAggDF.select(pandasUDF(*inputAggDF))

    explodedDF = outputAggDF.select(explode(*outputAggDF).alias("gapply"))
    # automatically retrieves nested schema column names
    return explodedDF.select("gapply.*")
