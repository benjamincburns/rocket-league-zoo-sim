from setuptools import setup, find_packages
from setuptools.command.install import install


__version__ = "1.0.0"

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()


class CustomInstall(install):
    def run(self):
        install.run(self)

setup(
    name='rlgym-sim',
    packages=find_packages(),
    version=__version__,
    description='A clone of RLGym for use with RocketSim in reinforcement learning projects.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lucas Emery, Matthew Allen, Zealan, and Ntud',
    url='https://github.com/AechPro/rocket-league-gym-sim',
    install_requires=[
        'gym>=0.17',
        'numpy>=1.19',
        'pyrocketsim @ git+https://github.com/uservar/pyrocketsim@dev'
    ],
    python_requires='>=3.7',
    cmdclass={'install': CustomInstall},
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'gym', 'reinforcement-learning', 'simulation']
)