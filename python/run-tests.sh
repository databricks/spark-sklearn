#!/usr/bin/env bash

# assumes run from python/ directory
if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:.

python pdspark/tests.py
