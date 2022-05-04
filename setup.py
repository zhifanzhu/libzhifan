from setuptools import setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

setup(
    name='libzhifan',
    packages=['libzhifan'],
    package_dir = {'libzhifan': 'libzhifan'},
    install_requires=[
        'numpy>=1.16.3',
        'matplotlib>=2.1.0',
        'pillow>=6.0.0',
    ],
    version='0.1',
)
