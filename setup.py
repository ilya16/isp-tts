from setuptools import setup, find_packages
from tts import __version__


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='isp-tts',
    description='Simple Text-to-Speech Synthesis Model',
    author='Ilya Borovik',
    author_email='ilya.borovik@skoltech.ru',
    version=__version__,
    packages=find_packages(),
    install_requires=requirements
)
