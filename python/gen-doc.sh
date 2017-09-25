#!/usr/bin/env bash

set -e

if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python$LIBS:$DIR

OUTPUT_DIR="$DIR/../docs/"

sphinx-build -E -a -j4 $DIR/doc/ $OUTPUT_DIR

echo $OUTPUT_DIR
