Welcome to the spark-sklearn Spark Package documentation!

This readme will walk you through navigating and building the spark-sklearn documentation, which is
included here with the source code.

## Generating the Documentation HTML

### Installing Dependencies

The spark-sklearn documentation is built with [Jekyll](http://jekyllrb.com), which
can be installed as follows:

    $ sudo gem install jekyll
    $ sudo gem install jekyll-redirect-from

On macOS, with the default Ruby, please install Jekyll with Bundler as
[instructed on offical website](https://jekyllrb.com/docs/quickstart/).
Otherwise the build script might fail to resolve dependencies.

    $ sudo gem install jekyll bundler
    $ sudo gem install jekyll-redirect-from

Install the python dependencies necessary for building the docs via (from project root):

    $ pip install -r python/requirements-docs.txt

### Building the Docs

Execute `jekyll build` from the `docs/` directory to compile the site.
When you run `jekyll build`, it will build (using Sphinx) the Python API
docs, copying them into the `docs` directory (and then also into the `_site` directory).

To serve the docs locally, run:

    # Serve content locally on port 4000
    $ jekyll serve --watch

Note that `SPARK_HOME` must be set to your local Spark installation in order to generate the docs.
To manually point to a specific `Spark` installation,
    $ SPARK_HOME=<your-path-to-spark-home> PRODUCTION=1 jekyll build
