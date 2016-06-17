
import uuid

from pyspark import SparkContext

# WARNING: These are private Spark APIs.
from pyspark.mllib.common import _py2java, _java2py

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

# 2.0.0-SNAPSHOT doesn't have pyspark.ml.common, so we fake its functionality here
# TODO once pyspark/ml/common.py is present in the snapshot, get rid of this painful hack.
def _hide_serde(sc):
    sc._jvm.OldSerDe = sc._jvm.SerDe
    sc._jvm.SerDe = sc._jvm.MLSerDe
def _recover_serde(sc):
    sc._jvm.SerDe = sc._jvm.OldSerDe
    del sc._jvm.OldSerDe

def _new_java_obj(sc, java_class, *args):
    """
    Construct a new Java object.
    """
    _hide_serde(sc)
    java_obj = _jvm()
    for name in java_class.split("."):
        java_obj = getattr(java_obj, name)
    java_args = [_py2java(sc, arg) for arg in args]
    _recover_serde(sc)
    return java_obj(*java_args)

def _call_java(sc, java_obj, name, *args):
    """
    Method copied from pyspark.ml.wrapper.  Uses private Spark APIs.
    """
    _hide_serde(sc)
    m = getattr(java_obj, name)
    sc = _wrap_sc(sc)
    java_args = [_py2java(sc, arg) for arg in args]
    res = _java2py(sc, m(*java_args))
    _recover_serde(sc)
    return res


def _randomUID(cls):
    """
    Generate a unique id for the object. The default implementation
    concatenates the class name, "_", and 12 random hex chars.
    """
    return cls.__name__ + "_" + uuid.uuid4().hex[12:]
