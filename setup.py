"""
Setup configuration for Applied Probability Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="applied-probability-framework",
    version="1.0.0",
    author="14ops",
    author_email="",
    description="Professional Monte Carlo simulation framework for probability analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/14ops/applied-probability-framework-me",
    project_urls={
        "Bug Tracker": "https://github.com/14ops/applied-probability-framework-me/issues",
        "Documentation": "https://github.com/14ops/applied-probability-framework-me/blob/main/docs/API_REFERENCE.md",
        "Source Code": "https://github.com/14ops/applied-probability-framework-me",
    },
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
            "pytest-timeout>=2.1.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
            "mypy>=0.950",
        ],
        "ml": [
            "tensorflow>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "apf=cli:main",
        ],
    },
    keywords="probability monte-carlo simulation bayesian-inference kelly-criterion statistics machine-learning",
    include_package_data=True,
    zip_safe=False,
)


