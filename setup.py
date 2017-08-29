from setuptools import setup 
#from Cython.Build import cythonize

setup(
    name="PyMUSE",
    version="0.1.4",
    description="Python software for handling VLT/MUSE data.",
    author="I. Pessa",
    license="MIT",
    author_email="ismael.pessa@gmail.com",
    url="https://github.com/ismaelpessa/Muse_Cube",
    packages=['PyMUSE'],
    install_requires=[
        'astropy',
        'scipy',
        'matplotlib',
        'aplpy',
        'h5py',
        'PyQT5'
        'pyregion',
        'linetools==0.2',
        'numpy'],
    dependenciy_links=[
        'https://github.com/linetools/linetools/tarball/master/#egg=linetools-0.2'])
