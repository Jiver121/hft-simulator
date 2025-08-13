"""
Setup configuration for HFT Order Book Simulator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hft-simulator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-Frequency Trading Order Book Simulator & Strategy Backtester",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hft-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.1.0",
        ],
        "advanced": [
            "ta-lib>=0.4.26",
            "pyfolio>=0.9.2",
            "zipline>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hft-simulator=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    keywords="hft high-frequency-trading order-book simulation backtesting quantitative-finance",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hft-simulator/issues",
        "Source": "https://github.com/yourusername/hft-simulator",
        "Documentation": "https://hft-simulator.readthedocs.io/",
    },
)