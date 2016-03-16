#!/usr/bin/env bash


# assumes run from python/ directory
if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#export PYSPARK_SUBMIT_ARGS="--jars pyspark-shell "

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:.

export PYTHONPATH=$PYTHONPATH:spark_sklearn

echo "TEST"
python -c "import scipy.sparse; print scipy.sparse"
echo "TEST DONE"

# Return on any failure
set -e

# Run test suites
exec nosetests -v --all-modules -w $DIR

# Horrible hack for spark 1.4: we manually remove some log lines to stay below the 4MB log limit on Travis.
# To remove when we ditch spark 1.4.
#exec nosetests -v --all-modules -w $DIR  2>&1 | grep -vE "INFO (PythonRunner|ContextCleaner|ShuffleBlockFetcherIterator|MapOutputTrackerMaster|TaskSetManager|Executor|MemoryStore|CacheManager|BlockManager|DAGScheduler|PythonRDD|TaskSchedulerImpl|ZippedPartitionsRDD2)"

