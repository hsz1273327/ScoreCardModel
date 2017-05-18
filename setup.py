import os
import sys
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='score_card_model',
    version='0.0.1',

    description='A sample orm for mongodb like mongoengine and peewee for asyncio',
    long_description=long_description,
    url='https://github.com/pypa/sampleproject',
    author='hsz',
    author_email='hsz1273327@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: orm',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='orm mongodb',

    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy','scipy','scikit-learn'],
    extras_require={
        'test': ['coverage'],
    }
)


if __name__ == '__main__':
    if sys.argv[-1] == 'test':
        os.system("python -m coverage run --source=score_card_model -m unittest discover -v -s test ")
        os.system("python -m coverage report")
        sys.exit()
