[![Documentation Status](https://readthedocs.org/projects/shesha/badge/?version=latest)](http://shesha.readthedocs.io/en/latest/?badge=latest) [![Anaconda-Server Badge](https://anaconda.org/compass/compass/badges/downloads.svg)](https://conda.anaconda.org/compass) [![Gitter](https://badges.gitter.im/ANR-COMPASS/community.svg)](https://gitter.im/ANR-COMPASS/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![CodeFactor](https://www.codefactor.io/repository/github/anr-compass/shesha/badge)](https://www.codefactor.io/repository/github/anr-compass/shesha)

Table of Contents
=================

- [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation of Anaconda/Miniconda](#installation-of-anacondaminiconda)
  - [Installation of COMPASS via conda](#installation-of-compass-via-conda)
  - [Installation of SHESHA package for COMPASS](#installation-of-shesha-package-for-compass)
  - [Test your installation](#test-your-installation)
  - [Run the simulation](#run-the-simulation)
  - [More documentation (maybe not fully up-to-date)](#more-documentation-maybe-not-fully-up-to-date)
  - [Questions?](#questions)

## Requirements

- Linux distribution with wget and git installed
- Nvidia GPU card with [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) >= 11.8 (Older versions could be available on request)

## Installation of Anaconda/Miniconda

COMPASS binaries, which contain the optimized GPU code, can be installed via Anaconda.
Then, you have to install Anaconda 3 or Miniconda 3 (python 3 is required).

We recommend Miniconda instead of Anaconda as it is much lighter, but it's up to you.

```bash
export CONDA_ROOT=$HOME/miniconda3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_ROOT
```

Don't forget to add your Miniconda or Anaconda directory to your PATH:

```bash
export PATH=$CONDA_ROOT/bin:$PATH
```

## Installation of COMPASS via conda
Once Miniconda is installed, installing the COMPASS binaries is easy :

```bash
conda install -c compass compass -y
```

**Note**: conda main channel is compiled with CUDA 12. For previous version please have a look to the [other channel](https://anaconda.org/compass/compass/) and post an issue and we will try to provide it.

This command line will also install dependencies in your conda environment.

## Installation of SHESHA package for COMPASS

First, you will need to set some environment variables:

```bash
export SHESHA_ROOT=$HOME/shesha
export PYTHONPATH=$SHESHA_ROOT:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
```

Finally, you can get the Shesha package of COMPASS. This python package is the user level of COMPASS. It also contains all the initialization functions.

```bash
git clone https://github.com/ANR-COMPASS/shesha.git $SHESHA_ROOT
```

## Test your installation

Once the installation is complete, verify that everything is working fine :
```bash
cd $SHESHA_ROOT/tests
./checkCompass.sh
```
This test will basically launch fast simulation test cases and it will print if those cases have been correctly initialised.

## Run the simulation

You are ready !
You can try it with one of our paramaters file:

```bash
cd $SHESHA_ROOT
ipython -i shesha/scripts/closed_loop.py data/par/par4bench/scao_sh_16x16_8pix.py
```

And if you want to launch the GUI:

```bash
cd $SHESHA_ROOT
ipython -i shesha/widgets/widget_ao.py
```

## More documentation (maybe not fully up-to-date)

Project GitHub pages with a detailed user manual : https://anr-compass.github.io/compass/

doc auto-generated from code: http://shesha.readthedocs.io

wiki page of the COMPASS project: https://projets-lesia.obspm.fr/projects/compass/wiki/Wiki

## Questions?

Please feel free to create an issue on Github for any questions and inquiries.

