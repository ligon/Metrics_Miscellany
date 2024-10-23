from distutils.core import setup
import sys

setup(
    name='Metrics_Miscellany',
    version='0.2.4',
    author='Ethan Ligon',
    author_email='ligon@berkeley.edu',
    packages=['metrics_miscellany',],
    license='Creative Commons Attribution-Noncommercial-ShareAlike 4.0 International license',
    description='Miscellaneous code for estimation involving pandas dataframes.',
    url='https://bitbucket.org/ligonresearch/metrics_miscellany',
    long_description=open('README.txt').read(),
    setup_requires = ['pytest_runner'],
    tests_require = ['pytest']
)
