#! /usr/bin/env python

import os
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """Functionals derived from the theory of entrpically regularized unbalanced optimal transport"""

version = None
with open(os.path.join('common', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'unbalanced-ot-functionals'
DESCRIPTION = descr
MAINTAINER = 'Thibault Séjourné'
MAINTAINER_EMAIL = 'thibault.sejourne@ens.fr'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/thibsej/unbalanced-ot-functionals.git'
VERSION = version
URL = 'https://github.com/thibsej/unbalanced-ot-functionals'


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          install_requires=[
          ],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=package_tree('unbalanced-ot-functionals'),
          )