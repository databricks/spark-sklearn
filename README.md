#PDSpark - Spark integration with PyData packages.

This package contains some tools to integrate the [Spark computing framework](http://spark.apache.org/) with popular [python data analysis tools](http://pydata.org/). Among other tools:
 - train and evaluate multiple scikit-learn models in parallel. It is an alternative to the [Joblib](https://pythonhosted.org/joblib/parallel.html)-based implementation included by default in scikit-learn.
 - convert Spark's Dataframes seemlessly into numpy `ndarray`s or spares matrices.
 - [experimental] distribute Scipy's sparse matrices as a dataset of sparse vectors.

  NOTE: This package is not a distributed implementation of scikit-learn on Spark.

## Licence

This package is released under the Apache 2.0 licence. See the LICENSE file.

## Installation

This package has the following requirements:
 - a recent version of scikit-learn. Version 0.17 has been tested, older versions may work too.
 - spark >= 1.6. Spark may be downloaded from the [spark official website](http://spark.apache.org/). In order to use PDSpark, you need to use the pyspark interpreter or another spark-compliant python interpreter (jupyter, zeppelin). See the [Spark guide](https://spark.apache.org/docs/0.9.0/python-programming-guide.html#interactive-use) for more details.
 - [nose](https://nose.readthedocs.org) (testing dependency only)

This package is available on PYPI:

	pip install XXX

If you want to use a developer version, you just need to make sure the `python/` subdirectory is in the `PYTHONPATH` when launching the pyspark interpreter:

	cd spark
	PYTHONPATH=$PYTHONPATH:/Users/tjhunter/work/spark-sklearn/python/ ./bin/pyspark

__Running tests__ You can directly run tests:

  cd python && ./run-tests.sh

This requires the environment variable `SPARK_HOME` to point to your local copy of spark.

## Example

Here is a simple example that runs a grid search in spark. See the [Installation] section on how to install PDSpark.


## Changelog

- 2015-12-10 First public release (0.1)

