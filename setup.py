from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(name='flowpy',
      version='0.4.2',
      description='Tools for working with optical flow',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://gitlab-research.centralesupelec.fr/2018seznecm/flowpy',
      author='Mickaël Seznec',
      author_email='mickael.seznec@centralesupelec.fr',
      license='MIT',
      packages=['flowpy'],
      install_requires=[
          'matplotlib',
          'numpy',
          'pypng',
          'scipy',
      ],
      test_requires=[
          'PIL',
      ],
      zip_safe=False)
