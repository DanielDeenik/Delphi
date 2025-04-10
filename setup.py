from setuptools import setup, find_packages

setup(
    name="delphi",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "google-cloud-bigquery",
        "pandas-gbq",
        "db-dtypes",
        "requests",
        "python-dotenv",
        "numba",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
            "pylint",
            "mypy",
        ],
        "yfinance": [
            "yfinance",
        ],
    },
    entry_points={
        "console_scripts": [
            "delphi-import=delphi.cli.import_command:import_data",
            "delphi-analyze=delphi.cli.analyze_command:analyze_data",
        ],
    },
    author="Delphi Team",
    author_email="info@delphi.com",
    description="A comprehensive financial analysis platform",
    keywords="finance, trading, analysis, bigquery, alpha vantage",
    url="https://github.com/yourusername/delphi",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
