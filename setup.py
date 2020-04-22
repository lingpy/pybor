#import distribute_setup
#distribute_setup.use_setuptools()

from setuptools import setup, find_packages,Extension
import codecs
# setup package name etc as a default
pkgname = 'mobor'


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
            'console_scripts': ['mobor=mobor.cli:main'],
        },
        author='John Miller',
        #author_email='list@shh.mpg.de',
        keywords='borrowing, language contact, sequence comparison'
        )
