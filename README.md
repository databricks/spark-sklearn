#Spark-sklearn - Spark integration with scikit-learn and numpy.

This package contains some tools to integrate the [Spark computing framework](http://spark.apache.org/) with the popular [scikit-learn machine library](http://scikit-learn.org/stable/). Among other tools:
 - train and evaluate multiple scikit-learn models in parallel. It is an alternative to the [Joblib](https://pythonhosted.org/joblib/parallel.html)-based implementation included by default in [scikit-learn](http://scikit-learn.org/stable/).
 - convert Spark's Dataframes seemlessly into numpy `ndarray`s or sparse matrices.
 - (experimental) distribute Scipy's sparse matrices as a dataset of sparse vectors.

  > NOTE: This package is not a distributed implementation of scikit-learn on Spark.

**Difference with the [sparkit-learn project](https://github.com/lensacom/sparkit-learn)** The sparkit-learn project aims at a comprehensive integration between Spark and scikit-learn. In particular, it adds some primitives to distribute numerical data using Spark, and it reimplements some of the most common algorithms found in scikit-learn. Spark-sklearn focuses on problems that have a small amount of data and that can be run in parallel.

## Licence

This package is released under the Apache 2.0 licence. See the LICENSE file.

## Installation

This package has the following requirements:
 - a recent version of scikit-learn. Version 0.17 has been tested, older versions may work too.
 - spark >= 1.5. Spark may be downloaded from the [spark official website](http://spark.apache.org/). In order to use Spark-sklearn, you need to use the pyspark interpreter or another spark-compliant python interpreter. See the [Spark guide](https://spark.apache.org/docs/1.5.2/programming-guide.html#overview) for more details.
 - [nose](https://nose.readthedocs.org) (testing dependency only)

This package is available on PYPI:

	pip install spark-sklearn

If you want to use a developer version, you just need to make sure the `python/` subdirectory is in the `PYTHONPATH` when launching the pyspark interpreter:

	cd spark
	PYTHONPATH=$PYTHONPATH:./python/ $SPARK_HOME/bin/pyspark

__Running tests__ You can directly run tests:

  cd python && ./run-tests.sh

This requires the environment variable `SPARK_HOME` to point to your local copy of Spark.

## Example

Here is a simple example that runs a grid search with Spark. See the [Installation] section on how to install Spark-sklearn.

```python
from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```

This classifier can be used as a drop-in replacement for any scikit-learn classifier. More larger datasets that do not fit in a single machine, we recommend the [machine learning algorithms implemented in Spark](https://spark.apache.org/docs/1.5.0/api/python/pyspark.mllib.html)

## Changelog

- 2015-12-10 First public release (0.1)

