from setuptools import setup, find_packages

setup(
    name='gym-env',
    version='0.0.2',
    packages=find_packages(),
    install_requires=['gymnasium', 'cosysairsim==3.2.0']
)
