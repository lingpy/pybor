import pathlib
import codecs
from setuptools import setup, find_packages, Extension

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_CONTENTS = (LOCAL_PATH / "README.md").read_text()

# Load requirements, so they are listed in a single place
REQUIREMENTS_PATH = LOCAL_PATH / "requirements.txt"
with open(REQUIREMENTS_PATH.as_posix()) as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

setup(
    author_email="list@shh.mpg.de",
    author="John Miller and Tiago Tresoldi and Johann-Mattis List",
    description="A Python library for monolingual borrowing detection.",
    extras_require={
        "examples": ["pyclts", "cldfbench", "pylexibank"],
        "tests": ["pytest", "pytest-cov"],
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords="borrowing, language contact, sequence comparison",
    license="Apache License 2.0",
    long_description_content_type="text/markdown",
    long_description=codecs.open("README.md", "r", "utf-8").read(),
    name="pybor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    url="https://github.com/lingpy/pybor/",
    version="1.0",  # remember to sync with __init__.py
    zip_safe=False,
)
