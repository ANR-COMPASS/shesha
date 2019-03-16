[![Documentation Status](https://readthedocs.org/projects/shesha/badge/?version=latest)](http://shesha.readthedocs.io/en/latest/?badge=latest) [![Anaconda-Server Badge](https://anaconda.org/compass/compass/badges/installer/conda.svg)](https://conda.anaconda.org/compass) [![Gitter](https://badges.gitter.im/ANR-COMPASS/community.svg)](https://gitter.im/ANR-COMPASS/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Table of Contents
=================

  * [Requirements](#requirements)
  * [Installation de COMPASS via conda](#installation-de-compass-via-conda)
  * [Installation de SHESHA package for COMPASS](#installation-de-shesha-package-for-compass)
  * [More documentation (maybe not fully up-to-date)](#more-documentation-maybe-not-fully-up-to-date)
  * [Questions?](#questions)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

## Requirements

Linux computer with CUDA >= 9.0

## Installation de COMPASS via conda

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH=/home/sevin/miniconda3/bin:$PATH
conda install -c compass compass -y
```

Note: conda main channel is compiled with CUDA 10.0, for previous version please use:

```bash
conda install -c compass/label/cuda92 compass
conda install -c compass/label/cuda91 compass
conda install -c compass/label/cuda90 compass
```

## Installation de SHESHA package for COMPASS

```bash
cd
git clone https://github.com/ANR-COMPASS/shesha.git
export SHESHA_ROOT=$HOME/shesha
export PYTHONPATH=$SHESHA_ROOT:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
cd $SHESHA_ROOT
ipython -i shesha/scripts/closed_loop.py data/par/par4bench/scao_sh_16x16_8pix.py
```

## More documentation (maybe not fully up-to-date)

Project GitHub pages with a detailed user manual : https://anr-compass.github.io/compass/

doc auto-generated from code: http://shesha.readthedocs.io

wiki page of the COMPASS project: https://projets-lesia.obspm.fr/projects/compass/wiki/Wiki

## Questions?

Please feel free to create an issue on Github for any questions and inquiries.

