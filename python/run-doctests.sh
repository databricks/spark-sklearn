#!/bin/bash
# Runs only the doctests. Additional flags are passed through to nose.

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ALL_MODULES=$(cd $DIR && echo 'import spark_sklearn, inspect; print(" ".join("spark_sklearn." + x[0] for x in inspect.getmembers(spark_sklearn, inspect.ismodule)))' | python)

$DIR/run-tests.sh $ALL_MODULES --with-doctest $@
