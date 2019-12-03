Deprecation
===========

This project is deprecated.
We now recommend using scikit-learn and `Joblib Apache Spark Backend <https://github.com/joblib/joblib-spark>`_
to distribute scikit-learn hyperparameter tuning tasks on a Spark cluster:

You need ``pyspark>=2.4.4`` and ``scikit-learn>=0.21`` to use Joblib Apache Spark Backend, which can be installed using ``pip``:

.. code:: bash

    pip install joblibspark

The following example shows how to distributed ``GridSearchCV`` on a Spark cluster using ``joblibspark``.
Same applies to ``RandomizedSearchCV``.

.. code:: python

    from sklearn import svm, datasets
    from sklearn.model_selection import GridSearchCV
    from joblibspark import register_spark
    from sklearn.utils import parallel_backend

    register_spark() # register spark backend

    iris = datasets.load_iris()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC(gamma='auto')

    clf = GridSearchCV(svr, parameters, cv=5)

    with parallel_backend('spark', n_jobs=3):
        clf.fit(iris.data, iris.target)


Scikit-learn integration package for Apache Spark
=================================================

This package contains some tools to integrate the `Spark computing framework <https://spark.apache.org/>`_
with the popular `scikit-learn machine library <https://scikit-learn.org/stable/>`_. Among other things, it can:

- train and evaluate multiple scikit-learn models in parallel. It is a distributed analog to the
  `multicore implementation <https://pythonhosted.org/joblib/parallel.html>`_ included by default in ``scikit-learn``
- convert Spark's Dataframes seamlessly into numpy ``ndarray`` or sparse matrices
- (experimental) distribute Scipy's sparse matrices as a dataset of sparse vectors

It focuses on problems that have a small amount of data and that can be run in parallel.
For small datasets, it distributes the search for estimator parameters (``GridSearchCV`` in scikit-learn),
using Spark. For datasets that do not fit in memory, we recommend using the `distributed implementation in
`Spark MLlib <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html>`_.

This package distributes simple tasks like grid-search cross-validation.
It does not distribute individual learning algorithms (unlike Spark MLlib).

Installation
------------

This package is available on PYPI:

::

	pip install spark-sklearn

This project is also available as `Spark package <https://spark-packages.org/package/databricks/spark-sklearn>`_.

The developer version has the following requirements:

- scikit-learn 0.18 or 0.19. Later versions may work, but tests currently are incompatible with 0.20.
- Spark >= 2.1.1. Spark may be downloaded from the `Spark website <https://spark.apache.org/>`_.
  In order to use this package, you need to use the pyspark interpreter or another Spark-compliant python
  interpreter. See the `Spark guide <https://spark.apache.org/docs/latest/programming-guide.html#overview>`_
  for more details.
- `nose <https://nose.readthedocs.org>`_ (testing dependency only)
- pandas, if using the pandas integration or testing. pandas==0.18 has been tested.

If you want to use a developer version, you just need to make sure the ``python/`` subdirectory is in the
``PYTHONPATH`` when launching the pyspark interpreter:

::

	PYTHONPATH=$PYTHONPATH:./python:$SPARK_HOME/bin/pyspark

You can directly run tests:

::

    cd python && ./run-tests.sh

This requires the environment variable ``SPARK_HOME`` to point to your local copy of Spark.

Example
-------

Here is a simple example that runs a grid search with Spark. See the `Installation <#installation>`_ section
on how to install the package.

.. code:: python

    from sklearn import svm, datasets
    from spark_sklearn import GridSearchCV
    iris = datasets.load_iris()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC(gamma='auto')
    clf = GridSearchCV(sc, svr, parameters)
    clf.fit(iris.data, iris.target)

This classifier can be used as a drop-in replacement for any scikit-learn classifier, with the same API.


Documentation
-------------

`API documentation <http://databricks.github.io/spark-sklearn-docs>`_ is currently hosted on Github pages. To
build the docs yourself, see the instructions in ``docs/``.

.. image:: https://travis-ci.org/databricks/spark-sklearn.svg?branch=master
    :target: https://travis-ci.org/databricks/spark-sklearn
