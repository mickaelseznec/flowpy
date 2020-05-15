from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(name='flowpy',
      version='0.4',
      description='Tools for working with optical flow',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/mickaelseznec/flowpy',
      author='Mickaël Seznec',
      author_email='flowpy@seznec.xyz',
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
