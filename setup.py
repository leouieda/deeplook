import os
from setuptools import setup, find_packages
# import versioneer

# VERSIONEER SETUP
# #############################################################################
# versioneer.VCS = 'git'
# versioneer.versionfile_source = 'deeplook/_version.py'
# versioneer.versionfile_build = 'deeplook/_version.py'
# versioneer.tag_prefix = 'v'
# versioneer.parentdir_prefix = '.'

# PACKAGE METADATA
# #############################################################################
NAME = 'deeplook'
FULLNAME = 'DeepLook'
DESCRIPTION = "Framework for building inverse problems"
AUTHOR = "Leonardo Uieda"
AUTHOR_EMAIL = 'leouieda@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
# VERSION = versioneer.get_version()
VERSION = 0.1
with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())
PACKAGES = find_packages(exclude=['doc', 'ci'])
LICENSE = "BSD License"
URL = "https://github.com/leouieda/deeplook"
PLATFORMS = "Any"
SCRIPTS = []
# PACKAGE_DATA = {'deeplook': [os.path.join('data', '*')]}
PACKAGE_DATA = {}
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5",
    "License :: OSI Approved :: {}".format(LICENSE),
]
KEYWORDS = 'inverse-problems geophysics'

# DEPENDENCIES
# #############################################################################
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'future',
]

if __name__ == '__main__':
    setup(name=NAME,
          fullname=FULLNAME,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          license=LICENSE,
          url=URL,
          platforms=PLATFORMS,
          scripts=SCRIPTS,
          packages=PACKAGES,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          install_requires=INSTALL_REQUIRES)
