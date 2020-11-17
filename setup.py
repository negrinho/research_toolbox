from setuptools import setup, find_packages

setup(name='research_toolbox',
      version='0.1',
      description='Utilities for reading and writing files, logging, and more.',
      long_description=open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      url='http://github.com/negrinho/research_toolbox',
      author='Renato Negrinho',
      author_email='renato.negrinho@gmail.com',
      license='MIT',
      packages=find_packages(include=["research_toolbox*"]),
      python_requires='>=3.6',
      install_requires=["paramiko", "numpy", "scipy", "psutil"],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ])