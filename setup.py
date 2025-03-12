from setuptools import setup, find_packages

setup(
    name="reflectance",
    version="1.0.0",
    description="Thin film characterization using reflectance spectroscopy",
    author="Hongrui Zhang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.21.0',
        'torch>=1.10.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'pyDOE>=0.3.8'
    ],
    python_requires=">=3.8",
) 