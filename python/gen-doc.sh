#!/usr/bin/env bash

set -e

if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to generate API docs.' >&2
    exit 1
fi

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python$LIBS:$DIR

OUTPUT_DIR="$DIR/doc_gen"

sphinx-build -E -a -j4 $DIR/doc/ $OUTPUT_DIR
