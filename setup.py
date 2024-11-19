# setup.py
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ccmMul",
    version="1.0.0",
    author="Sharath Sathish",
    author_email="your.email@example.com",
    description="Multivariate Convergent Cross Mapping (CCM) analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SharathSPhD/ccmMul",
    project_urls={
        "Bug Tracker": "https://github.com/SharathSPhD/ccmMul/issues",
        "Documentation": "https://ccmMul.readthedocs.io/",
        "Source Code": "https://github.com/SharathSPhD/ccmMul",
    },
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-xdist>=3.3.0',
            'pytest-timeout>=2.1.0',
            'black>=23.0.0',
            'pylint>=2.17.0',
            'mypy>=1.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
            'tox>=4.5.0',
            'pre-commit>=3.2.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'sphinx-autodoc-typehints>=1.22.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ccmMul=src.main:main',
        ],
    },
    package_data={
        'src': ['py.typed'],
    },
)