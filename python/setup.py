import codecs
import os

from setuptools import setup


###############################################################################

PACKAGES = ["pdspark"]
KEYWORDS = ["spark", "scikit-learn", "distributed computing", "machine learning"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]
INSTALL_REQUIRES = ["scikit-learn >= 0.17"]

###############################################################################

HERE = os.path.abspath(os.path.dirname(__file__) + "../")


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


if __name__ == "__main__":
  setup(
    name="spark-sklearn",
    description="Integrations tools for running scikit-learn on Spark",
    license="Apache 2.0",
    url="missing",
    version="0.1.0",
    author="Joseph Bradley",
    author_email="joseph@databricks.com",
    maintainer="Joseph Bradley",
    maintainer_email="joseph@databricks.com",
    keywords=KEYWORDS,
    long_description=read("README.md"),
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
  )
