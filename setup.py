#! /usr/bin/env python

from setuptools import setup
setup(
    name="unbalancedot",
    distname="",
    version='0.0.1',
    description="Functionals derived from the theory of entropically "
                "regularized unbalanced optimal transport ",
    author='Thibault Sejourne',
    author_email='thibault.sejourne@ens.fr',
    url='https://github.com/thibsej/unbalanced-ot-functionals',
    packages=['unbalancedot', 'unbalancedot.tests'],
    install_requires=[
              'numpy',
              'torch',
            'scipy',
            'pytest'
          ],
    license="MIT",
)
