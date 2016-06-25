
import uuid

from pyspark import SparkContext
from pyspark.sql import SparkSession

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

def createLocalSparkSession(appName="spark-sklearn"):
    """Generates a :class:`SparkSession` utilizing all local cores
    with the progress bar disabled but otherwise default config."""
    return SparkSession.builder \
                       .master("local[*]") \
                       .appName(appName) \
                       .config("spark.ui.showConsoleProgress", "false") \
                       .getOrCreate()
