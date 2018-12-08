# Generating the Documentation HTML

## Installing Dependencies

The spark-sklearn documentation is built with [Jekyll](https://jekyllrb.com), which
can be installed as follows:

    sudo gem install jekyll

On macOS, with the default Ruby, please install Jekyll with Bundler as
[instructed on official website](https://jekyllrb.com/docs/quickstart/).
Otherwise the build script might fail to resolve dependencies.

    sudo gem install jekyll bundler

Install the python dependencies necessary for building the docs via (from project root):

    pip install -r python/requirements-docs.txt

## Building the Docs

Execute `jekyll build` from the `docs/` directory to compile the site.
When you run `jekyll build`, it will build (using Sphinx) the Python API
docs, copying them into the `docs` directory (and then also into the `_site` directory).

To serve the docs locally on port 4000, run:

    SPARK_HOME=<your-path-to-spark-home> jekyll serve --watch
