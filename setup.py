import sys
from os.path import join, dirname, abspath
from setuptools import setup, Extension

from distutils.command import build_ext

def get_export_symbols(self, ext):
    """From https://bugs.python.org/issue35893"""
    parts = ext.name.split(".")
    # print('parts', parts)
    if parts[-1] == "__init__":
        initfunc_name = "PyInit_" + parts[-2]
    else:
        initfunc_name = "PyInit_" + parts[-1]

build_ext.build_ext.get_export_symbols = get_export_symbols


# Require pytest-runner only when running tests
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup_requires = pytest_runner

libnrlmsise00 = Extension(
    'MCEq.geometry.nrlmsise00.libnrlmsise00',
    sources=[
        join('MCEq/geometry/nrlmsise00', sf)
        for sf in ['nrlmsise-00_data.c', 'nrlmsise-00.c']
    ],
    include_dirs=['MCEq/geometry/nrlmsise00'])
    
libcorsikaatm = Extension(
    'MCEq.geometry.corsikaatm.libcorsikaatm',
    sources=['MCEq/geometry/corsikaatm/corsikaatm.c'])


# This method is adopted from iMinuit https://github.com/scikit-hep/iminuit
# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/en/latest/development.html#single-sourcing-the-version-across-setup-py-and-your-project
# This is one of the recommended methods that works in Python 2 and 3:
def get_version():
    version = {}
    with open("MCEq/version.py") as fp:
        exec (fp.read(), version)
    return version['__version__']


__version__ = get_version()

this_directory = abspath(dirname(__file__))
if sys.version_info.major == 3:
    with open(join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
else:
    with open(join(this_directory, 'README.md')) as f:
        long_description = f.read()

skip_marker = "# MCEq"
long_description = long_description[long_description.index(skip_marker) :].lstrip()

setup(
    name='MCEq',
    version=__version__,
    description='Numerical cascade equation solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anatoli Fedynitch',
    author_email='afedynitch@gmail.com',
    license='BSD 3-Clause License',
    url='https://github.com/afedynitch/MCEq',
    packages=['MCEq', 'MCEq.tests', 'MCEq.geometry',
        'MCEq.geometry.nrlmsise00', 'MCEq.geometry.corsikaatm'],
    setup_requires=[] + pytest_runner,
    package_data={
        'MCEq': ['data/README.md', "geometry/nrlmsise00/nrlmsise-00.h"],
    },
    install_requires=[
        'six',
        'h5py',
        'particletools',
        'crflux>1.0.4',
        'scipy',
        'numpy',
        'tqdm',
        'requests'
    ],
    py_modules=['mceq_config'],
    ext_modules=[libnrlmsise00, libcorsikaatm],
    extras_require={
        'MKL': ['mkl==2020.0'],
        'CUDA': ['cupy-cuda112==9.2.0']
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License'
    ])
