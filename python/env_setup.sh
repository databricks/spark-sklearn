#!/usr/bin/env bash
# Sets up enviornment from test. Should be sourced in other scripts.

if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:.

export PYTHONPATH=$PYTHONPATH:spark_sklearn

# Use the miniconda environment:
export PYTHONPATH=$PYTHONPATH:/home/travis/miniconda/envs/test-environment/lib/python2.7/site-packages/
