#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for the Delphi Trading Intelligence System."""

import os
import re
from setuptools import setup, find_packages

# Get requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="delphi",
    version="0.3.0",
    packages=find_packages(include=['trading_ai', 'trading_ai.*']),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.10.0",
            "isort>=5.10.0",
            "flake8>=6.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.1.0",
        ],
        "yfinance": [
            "yfinance>=0.2.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "delphi-import=trading_ai.cli.import_cli:main",
            "delphi-analyze=trading_ai.cli.analyze_cli:main",
            "delphi-dashboard=trading_ai.cli.dashboard_cli:main",
            "delphi-notebook=trading_ai.cli.notebook_cli:main",
        ],
    },
    author="Delphi Team",
    author_email="info@delphitrading.ai",
    description="A cloud-native, AI-powered trading intelligence system",
    long_description=open('README.md', 'r', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    keywords="finance, trading, analysis, bigquery, alpha vantage, time series, ai, machine learning",
    url="https://github.com/delphitrading/delphi",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
)
