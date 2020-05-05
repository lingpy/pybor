import pathlib
import codecs
from setuptools import setup, find_packages, Extension

# setup package name etc as a default
pkgname = 'pybor'

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_CONTENTS = (LOCAL_PATH / "README.md").read_text()

# Load requirements, so they are listed in a single place
REQUIREMENTS_PATH = LOCAL_PATH / "requirements.txt"
with open(REQUIREMENTS_PATH.as_posix()) as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

setup(
        name=pkgname,
        description="A Python library for monolingual borrowing detection.",
        version='0.1.1',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        zip_safe=False,
        license="GPL",
        include_package_data=True,
        install_requires=['cldfbench', 'pyclts', 'lingpy', 'matplotlib'],
        url='https://github.com/lingpy/monolingual-borrowing-detection/',
        long_description=codecs.open('README.md', 'r', 'utf-8').read(),
        long_description_content_type='text/markdown',
        entry_points={
            'console_scripts': ['pybor=pybor.cli:main'],
        },
        author='John Miller and Tiago Tresoldi and Johann-Mattis List',
        author_email='list@shh.mpg.de',
        keywords='borrowing, language contact, sequence comparison'
