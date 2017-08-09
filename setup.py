#!/usr/bin/env python

from distutils.core import setup

setup(
    name='MCEq',
    version='rc1',
    description='Numerical cascade equation solver',
    author='Anatoli Fedynitch',
    author_email='afedynitch@gmail.com',
    url='https://github.com/afedynitch/MCEq.git',
    packages=[
        'MCEq', 'CRFluxModels', 'ParticleDataTool', 'Python-NRLMSISE-00'
    ],
    py_modules=['mceq_config'],
    requires=['numpy', 'scipy', 'numba', 'matplotlib', 'jupyter', 'ctypes'])