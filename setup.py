#!/usr/bin/env python2

from distutils.core import setup

setup(
    name='bmtools',
    description='Biomechanics tools',
    version='1.0.1',
    packages=['bmtools'],
    include_package_data=True,
    url='https://github.com/gepetto/bmtools',
    author='Galo MALDONADO',
    author_email='galo.maldonado@laas.fr',
    license='GPL',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
)
