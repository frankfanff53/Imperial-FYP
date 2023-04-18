"""Install Time Series Analysis Package."""

import setuptools
import os

# Get the long description from the README file.
_LONG_DESCRIPTION = "tackling_crohns_disease_2023"
if os.path.exists("README.md"):
    with open('README.md') as fp:
        _LONG_DESCRIPTION = fp.read()

extras = {}

extras["style"] = ["autoflake", "black", "flake8", "isort"]
extras["test"] = ["pytest", "pytest-cov"]

extras["dev"] = extras["style"] + extras["test"]

# Building list from requirements.txt
root_folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = root_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name='tackling_crohns_disease_2023',
    version='1.0.0',
    description='tackling_crohns_disease_2023',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Feifan Fan',
    packages=setuptools.find_packages(),
    package_data={},
    scripts=[],
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
    ]
)
