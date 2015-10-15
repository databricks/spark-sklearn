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
    Class for converting between scikit-learn and Spark ML models
    """

    def __init__(self, sc):
        """
        :param sc: SparkContext
        """
        self.sc = sc
        # For conversions sklearn -> Spark
        self._skl2spark_classes = {
            SKL_LogisticRegression :
                ClassNames("org.apache.spark.ml.classification.LogisticRegressionModel",
                           LogisticRegressionModel),
            SKL_LinearRegression :
                ClassNames("org.apache.spark.ml.regression.LinearRegressionModel",
                           LinearRegressionModel)
        }
        self._supported_skl_types = self._skl2spark_classes.keys()
        # For conversions Spark -> sklearn
        self._spark2skl_classes =\
            dict([(self._skl2spark_classes[skl].py, skl) for skl in self._skl2spark_classes.keys()])
        self._supported_spark_types = [x.py for x in self._skl2spark_classes.values()]

    def toSpark(self, model):
        """
        Convert a scikit-learn model to a Spark ML model from the Pipelines API (spark.ml).
        Currently supported models:
          sklearn.linear_model.LogisticRegression (binary classification only, not multiclass)
          sklearn.linear_model.LinearRegression
        :param model: scikit-learn model
        :return: Spark ML model with equivalent predictive behavior.
                 Currently, parameters or arguments for training are not copied.
        """
        if isinstance(model, SKL_LogisticRegression) or isinstance(model, SKL_LinearRegression):
            return self._toSparkGLM(model)
        else:
            supported_types = map(lambda t: type(t), self._supported_skl_types)
            raise ValueError("Converter.toSpark cannot convert type: %s.  Supported types: %s" %
                             (type(model), ", ".join(supported_types)))

    def _toSparkGLM(self, model):
        """ Private method for converting a GLM to a Spark model
        TODO: Add model parameters as well.
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

    def toSKLearn(self, model):
        """
        Convert a Spark MLlib model from the Pipelines API (spark.ml) to a scikit-learn model.
        Currently supported models:
          pyspark.ml.classification.LogisticRegressionModel
          pyspark.ml.regression.LinearRegressionModel
        :param model: Spark ML model
        :return: scikit-learn model with equivalent predictive behavior.
                 Currently, parameters or arguments for training are not copied.
        """
        if isinstance(model, LogisticRegressionModel) or isinstance(model, LinearRegressionModel):
            return self._toSKLGLM(model)
        else:
            supported_types = map(lambda t: type(t), self._supported_spark_types)
            raise ValueError("Converter.toSKLearn cannot convert type: %s.  Supported types: %s" %
                             (type(model), ", ".join(supported_types)))

    def _toSKLGLM(self, model):
        """ Private method for converting a GLM to a scikit-learn model
        TODO: Add model parameters as well.
        """
        py_cls = type(model)
        skl_cls = self._spark2skl_classes[py_cls]
        intercept = model.intercept
        weights = model.weights
        skl = skl_cls()
        skl.intercept_ = numpy.float64(intercept)
        skl.coef_ = weights.toArray()
        return skl
