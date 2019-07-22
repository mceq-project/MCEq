import sys
from os.path import join
from setuptools import setup, Extension

# Require pytest-runner only when running tests
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup_requires = pytest_runner

libnrlmsise00 = Extension(
    'MCEq/nrlmsise00/_libnrlmsise00',
    sources=[
        join('MCEq/nrlmsise00', sf)
        for sf in ['nrlmsise-00_data.c', 'nrlmsise-00.c']
    ],
    include_dirs=['MCEq/nrlmsise00'])


# This method is adopted from iMinuit https://github.com/scikit-hep/iminuit
# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/en/latest/development.html#single-sourcing-the-version-across-setup-py-and-your-project
# This is one of the recommended methods that works in Python 2 and 3:
def get_version():
    version = {}
    with open("MCEq/info.py") as fp:
        exec (fp.read(), version)
    return version['__version__']


__version__ = get_version()

setup(
    name='MCEq',
    version=__version__,
    description='Numerical cascade equation solver',
    author='Anatoli Fedynitch',
    author_email='afedynitch@gmail.com',
    license='BSD 3-Clause License',
    url='https://github.com/afedynitch/MCEq',
    packages=['MCEq', 'MCEq.nrlmsise00', 'MCEq.geometry'],
    setup_requires=[] + pytest_runner,
    package_dir={
        'MCEq': 'MCEq',
        'MCEq.geometry': 'MCEq/geometry',
        'MCEq.nrlmsise00': 'MCEq/nrlmsise00'
    },
    package_data={
        'MCEq': ['data/README.md'],
    },
    install_requires=[
        'six>=1.12.0',
        'h5py>=2.9.0',
        'particletools>=1.1.3',
        'crflux>=1.0.1',
        'scipy>=1.2.1',
        'numpy>=1.16.2',
        'numba>=0.43.1',
        'mock>=3.0.5'
    ],
    py_modules=['mceq_config'],
    requires=[
        'numpy', 'scipy', 'numba', 'mkl', 'particletools', 'crflux', 'h5py'
    ],
    ext_modules=[libnrlmsise00],
    extras_require={
        'MKL': ['mkl>=2019.1'],
        'CUDA': ['cupy>=5.1.0']
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License'
    ])
