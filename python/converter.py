"""
Class for converting between scikit-learn models and PySpark ML models
"""

from collections import namedtuple

import numpy

from sklearn.linear_model import LogisticRegression as SKL_LogisticRegression
from sklearn.linear_model import LinearRegression as SKL_LinearRegression

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.mllib.linalg import Vectors

from util import _new_java_obj, _randomUID

ClassNames = namedtuple('ClassNames', ['jvm', 'py'])

class Converter(object):
    """
    TODO
    """

    def __init__(self, sc):
        self.sc = sc
        self._skl2spark_classes = {
            SKL_LogisticRegression :
                ClassNames("org.apache.spark.ml.classification.LogisticRegressionModel",
                           LogisticRegressionModel),
            SKL_LinearRegression :
                ClassNames("org.apache.spark.ml.regression.LinearRegressionModel",
                           LinearRegressionModel)
        }
        self._supported_skl_types = self._skl2spark_classes.keys()

    def toSpark(self, model):
        if isinstance(model, SKL_LogisticRegression) or isinstance(model, SKL_LinearRegression):
            return self._toSparkGLM(model)
        else:
            supported_types = map(lambda t: type(t), self._supported_skl_types)
            raise ValueError("Converter.toSpark cannot convert type: %s.  Supported types: %s" %
                             (type(model), ", ".join(supported_types)))

    def _toSparkGLM(self, model):
        """ Private method for converting a GLM to a Spark model
        """
        skl_cls = type(model)
        py_cls = self._skl2spark_classes[skl_cls].py
        jvm_cls_name = self._skl2spark_classes[skl_cls].jvm
        intercept = model.intercept_
        weights = model.coef_
        if len(numpy.shape(weights)) == 1\
                or (len(numpy.shape(weights)) == 2 and numpy.shape(weights)[0] == 1):
            # Binary classification
            uid = _randomUID(skl_cls)
            _java_model = _new_java_obj(self.sc, jvm_cls_name, uid, Vectors.dense(weights), float(intercept))
            return py_cls(_java_model)
        elif len(numpy.shape(weights)) == 2 and skl_cls == SKL_LogisticRegression:
            # Multiclass label
            raise ValueError("Converter.toSpark cannot convert a multiclass sklearn Logistic" +
                             " Regression model to Spark because Spark does not yet support" +
                             " multiclass.  Given model is for %d classes." %
                             numpy.shape(weights)[0])
        else:
            raise Exception("Converter.toSpark experienced unknown error when trying to convert" +
                            " a model of type: " + type(model) + "  " + len(numpy.shape(weights)))
