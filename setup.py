from setuptools import setup 
#from Cython.Build import cythonize

setup(
    name="PyMUSE",
    version="0.4.4",
    description="Python software for handling VLT/MUSE data.",
    author="I. Pessa",
    license="MIT",
    author_email="ismael.pessa@gmail.com",
    url="https://github.com/ismaelpessa/PyMUSE",
    packages=['PyMUSE'],
    install_requires=[
        'astropy',
        'scipy',
        'matplotlib',
        'aplpy',
        'h5py',
        'pyregion',
        'numpy',
        'mpdaf'
          ])
