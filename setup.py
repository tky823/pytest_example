from setuptools import setup, find_packages

setup(
    name="pybss",
    version="0.0.0",
    url="https://github.com/tky823/pytest_example",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy",
    ]
)