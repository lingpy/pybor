import pathlib
from setuptools import setup, find_packages, Extension

# setup package name etc as a default
pkgname = "mobor"

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_CONTENTS = (LOCAL_PATH / "README.md").read_text()

# Load requirements, so they are listed in a single place
REQUIREMENTS_PATH = LOCAL_PATH / "requirements.txt"
with open(REQUIREMENTS_PATH.as_posix()) as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

setup(
    author="John Miller",
    classifiers=[
        "License :: OSI Approved :: GPL License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
    description="A Python library for monolingual borrowing detection.",
    entry_points={"console_scripts": ["mobor=mobor.cli:main"]},
    include_package_data=True,
    install_requires=install_requires,
    keywords="borrowing, language contact, sequence comparison",
    license="GPL",
    long_description_content_type="text/markdown",
    long_description=README_CONTENTS,
    name=pkgname,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    test_suite="tests",
    tests_require=[],
    url="https://github.com/lingpy/monolingual-borrowing-detection/",
    version="0.1.1",
    zip_safe=False,
)
