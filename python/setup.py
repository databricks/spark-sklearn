import codecs
import os

from setuptools import setup
# See this web page for explanations:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
PACKAGES = ["spark_sklearn"]
KEYWORDS = ["spark", "scikit-learn", "distributed computing", "machine learning"]
CLASSIFIERS = [
	"Programming Language :: Python :: 2.6",
	"Programming Language :: Python :: 2.7",
	"Programming Language :: Python :: 3",
	"Programming Language :: Python :: 3.2",
	"Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]
INSTALL_REQUIRES = ["scikit-learn >= 0.17"]

# Project root
ROOT = os.path.abspath(os.getcwd() + "/")


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(ROOT, *parts), "rb", "utf-8") as f:
        return f.read()

setup(
	name="spark-sklearn",
	description="Integration tools for running scikit-learn on Spark",
	license="Apache 2.0",
	url="https://github.com/databricks/spark-sklearn",
	version="0.2.0",
	author="Joseph Bradley",
	author_email="joseph@databricks.com",
	maintainer="Tim Hunter",
	maintainer_email="timhunter@databricks.com",
	keywords=KEYWORDS,
	long_description=read("README.md"),
	packages=PACKAGES,
	classifiers=CLASSIFIERS,
	zip_safe=False,
	install_requires=INSTALL_REQUIRES
)
