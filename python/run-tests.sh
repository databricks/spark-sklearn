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

# Use the miniconda environment:
export PYTHONPATH=$PYTHONPATH:/home/travis/miniconda/envs/test-environment/lib/python2.7/site-packages/

# Return on any failure
set -e

if [ "$#" = 0 ]; then
    ARGS="--all-modules"
else
    ARGS="$@"
fi
exec nosetests -v "$ARGS" -w $DIR
