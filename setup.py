from setuptools import setup, find_packages

with open('LICENSE') as f:
    lic = f.read()

with open('README.md') as f:
    readme = f.read()

setup(author="quantfaf",
      author_email="quantfaf@gmail.com",
      description="A simple portfolio rebalancing package.",
      name="portfoliorebalancer",
      version="0.1.0",
      license=lic,
      readme=readme,
      url="https://github.com/quantfaf/portfoliorebalancer",
      install_requires=['pandas>=1.3.4', 'numpy>=1.21.1', 'scipy>=1.7.3', 'matplotlib>=3.5.1', 'seaborn>=0.11.1'],
      packages=find_packages(include="portfoliorebalancer"))
