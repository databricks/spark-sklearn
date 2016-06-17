echo "Downloading Spark if necessary"
echo "Spark version = $SPARK_VERSION"
echo "Spark build = $SPARK_BUILD"
echo "Spark build URL = $SPARK_BUILD_URL"
mkdir -p $HOME/.cache/spark-versions
tarfilename="$HOME/.cache/spark-versions/$SPARK_BUILD.tgz"
dirname="$HOME/.cache/spark-versions/$SPARK_BUILD"
if ! [ -d $dirname ]; then
    echo "Missing $dirname, downloading archive"
    echo `which curl`
    curl "$SPARK_BUILD_URL" > $tarfilename
    tar xvf $tarfilename --directory $HOME/.cache/spark-versions > /dev/null
    echo "Content of directory:"
    ls -la $HOME/.cache/spark-versions/
else
    echo "Skipping download - found spark dir $dirname"
fi
