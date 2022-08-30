from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name = 'spych',
  packages = ['spych'],
  version = '0.0.5',
  license='MIT',
  description = 'Python wrapper for the deepspeech library',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Connor Makowski',
  author_email = 'connor.m.makowski@gmail.com',
  url = 'https://github.com/connor-makowski/spych',
  download_url = 'https://github.com/connor-makowski/spych/dist/spych-0.0.5.tar.gz',
  keywords = ['function', 'voice', 'machine','learning'],
  install_requires=[
    'deepspeech==0.9.3'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
