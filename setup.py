try:  
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name='deep_learning_layer_calculator',
    version='0.1.0',
    description='Calculates layers for deep learning based on user specified input dimensions, output dimensions, and architecture.',
    author='Makenzie Brian',
    author_email='brianm@oregonstate.edu',
    url='',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
    ],
    license='MIT',
    python_requires='>=2',
    zip_safe=False,
    packages=['deep_learning_layer_calculator', 'deep_learning_layer_calculator.tests'],

    package_dir={
        'deep_learning_layer_calculator': 'deep_learning_layer_calculator',
        'deep_learning_layer_calculator.tests': 'deep_learning_layer_calculator/tests',
        },
    include_package_data=True,

)
