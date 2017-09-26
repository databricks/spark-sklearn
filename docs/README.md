Welcome to the spark-sklearn Spark Package documentation!

This readme will walk you through navigating and building the spark-sklearn documentation, which is
included here with the source code.

Read on to learn more about viewing documentation in plain text (i.e., markdown) or building the
documentation yourself.

## Generating the Documentation HTML

In this directory you will find textfiles formatted using Markdown, with an ".md" suffix. You can
read those text files directly if you want. Start with index.md.

The markdown code can be compiled to HTML using the [Jekyll tool](http://jekyllrb.com).
`Jekyll` and a few dependencies must be installed for this to work. We recommend
installing via the Ruby Gem dependency manager. Since the exact HTML output
varies between versions of Jekyll and its dependencies, we list specific versions here
in some cases:

    $ sudo gem install jekyll
    $ sudo gem install jekyll-redirect-from

On macOS, with the default Ruby, please install Jekyll with Bundler as [instructed on offical website](https://jekyllrb.com/docs/quickstart/). Otherwise the build script might fail to resolve dependencies.

    $ sudo gem install jekyll bundler
    $ sudo gem install jekyll-redirect-from

Install the python dependencies necessary for building the docs via:

    $ pip install -r requirements-docs.txt

Execute `jekyll build` from the `docs/` directory to compile the site.
When you run `jekyll build` in the `docs` directory, it will build (using Sphinx) the Python API
docs, copying them into the `docs` directory (and then also into the `_site` directory).

To serve the docs
locally, run:

    # Serve content locally on port 4000
    $ jekyll serve --watch

Note that `SPARK_HOME` must be set to your local Spark installation in order to generate the docs.
To manually point to a specific `Spark` installation,
    $ SPARK_HOME=<your-path-to-spark-home> PRODUCTION=1 jekyll build
