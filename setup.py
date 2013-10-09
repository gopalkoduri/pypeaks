from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pypeaks',
      version='0.2.5',
      description='Python module with different methods to identify peaks from data like histograms and time-series data',
      long_description=readme(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords='python peaks histogram time-series maxima minima',
      url='https://github.com/gopalkoduri/pypeaks',
      author='Gopala Krishna Koduri',
      author_email='gopala.koduri@gmail.com',
      license='GNU Affero GPL v3',
      packages=['pypeaks'],
      install_requires=[
          'numpy',
          'matplotlib',
      ],
      zip_safe=False)
