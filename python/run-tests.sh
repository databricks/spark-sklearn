#!/usr/bin/env bash
# Runs both doctests and unit tests by default, otherwise hands arguments over to nose.

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/env_setup.sh

if [ "$#" = 0 ]; then
    ARGS="--nologcapture --all-modules --verbose --with-doctest"
else
    ARGS="$@"
fi
exec nosetests $ARGS --where $DIR
