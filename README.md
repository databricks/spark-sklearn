#spark-sklearn

Integration of Spark ML pipeline API with scikit-learn.

This package lets you train and evaluate multiple scikit-learn models in parallel, using the [spark computing framework](http://spark.apache.org/) for scheduling. 
It is an alternative to the [Joblib](https://pythonhosted.org/joblib/parallel.html)-based implementation included by default in scikit-learn.

  NOTE: This package is not a distributed implementation of scikit-learn on Spark.

## Installation

This package has the following requirements:
 - scikit-learn (version 0.17 has been tested, older versions may work too)
 - spark >= 1.6 Spark may be downloaded from the [spark official website](http://spark.apache.org/). In order to use spark-skleran, you need to use the pyspark interpreter or another spark-compliant python interpreter (jupyter, zeppelin).

This package is available on PYPI:

	pip install XXX

If you want to use a developer version, you just need to make sure is in the PYTHONPATH when launching the pyspark interpreter:

	cd spark
	PYTHONPATH=$PYTHONPATH:/Users/tjhunter/work/spark-sklearn/python/ ./bin/pyspark

__Running tests__ You can directly run tests:

  ./python/run-tests.sh

This requires the environment variable `SPARK_HOME` to point to your local copy of spark.

## Example

Here is a simple example that runs a grid search in spark. See the [Installation] section on how to install sp-learn.


## Changelog

- 2015-12-10 First public release (0.1)

