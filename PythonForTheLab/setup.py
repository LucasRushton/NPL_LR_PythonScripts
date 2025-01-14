from setuptools import setup, find_packages

setup(
    name = 'PFTL',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['py4lab=PFTL.start:start',]
        },
    install_requires=[
        'PyQt5',
        'PyYAML',
        'pyserial',
        'numpy',
        'pyvisa',
        'matplotlib',
        'pyqtgraph',
        ],
)