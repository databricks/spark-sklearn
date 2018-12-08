# Scikit-learn integration package for Apache Spark

This package contains some tools to integrate the [Spark computing framework](https://spark.apache.org/) with the popular [scikit-learn machine library](https://scikit-learn.org/stable/). Among other tools:
- train and evaluate multiple scikit-learn models in parallel. It is a distributed analog to the [multicore implementation](https://pythonhosted.org/joblib/parallel.html) included by default in [scikit-learn](https://scikit-learn.org/stable/).
- convert Spark's Dataframes seamlessly into numpy `ndarray`s or sparse matrices.
- (experimental) distribute Scipy's sparse matrices as a dataset of sparse vectors.

It focuses on problems that have a small amount of data and that can be run in parallel.
- for small datasets, it distributes the search for estimator parameters (`GridSearchCV` in scikit-learn), using Spark,
- for datasets that do not fit in memory, we recommend using the [distributed implementation in Spark MLlib](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html).
This package distributes simple tasks like grid-search cross-validation. It does not distribute individual learning algorithms (unlike Spark MLlib).

## Installation

This package is available on PYPI:

	pip install spark-sklearn

This project is also available as as [Spark package](https://spark-packages.org/package/databricks/spark-sklearn).

The developer version has the following requirements:
 - a recent release of scikit-learn. Releases 0.18.1, 0.19.0 have been tested, older versions may work too.
 - Spark >= 2.1.1. Spark may be downloaded from the [Spark official website](https://spark.apache.org/). In order to use this package, you need to use the pyspark interpreter or another Spark-compliant python interpreter. See the [Spark guide](https://spark.apache.org/docs/latest/programming-guide.html#overview) for more details.
 - [nose](https://nose.readthedocs.org) (testing dependency only)
 - Pandas, if using the Pandas integration or testing. Pandas==0.18 has been tested.

If you want to use a developer version, you just need to make sure the `python/` subdirectory is in the `PYTHONPATH` when launching the pyspark interpreter:

	PYTHONPATH=$PYTHONPATH:./python:$SPARK_HOME/bin/pyspark

__Running tests__ You can directly run tests:

    cd python && ./run-tests.sh

This requires the environment variable `SPARK_HOME` to point to your local copy of Spark.

## Example

Here is a simple example that runs a grid search with Spark. See the [Installation](#installation) section on how to install the package.

```python
from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(sc, svr, parameters)
clf.fit(iris.data, iris.target)
```

This classifier can be used as a drop-in replacement for any scikit-learn classifier, with the same API.

## Documentation

[API documentation](http://databricks.github.io/spark-sklearn-docs) is currently hosted on Github pages. To
build the docs yourself, see the instructions in [docs/README.md](https://github.com/databricks/spark-sklearn/tree/master/docs).

[![Build Status](https://travis-ci.org/databricks/spark-sklearn.svg?branch=master)](https://travis-ci.org/databricks/spark-sklearn)