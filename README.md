Installation de COMPASS via conda

```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH=/home/sevin/miniconda3/bin:$PATH
conda install -c compass compass
git clone -b py3 git@gitlab.obspm.fr:compass/shesha.git
export SHESHA_ROOT=/home/sevin/shesha
export PYTHONPATH=$SHESHA_ROOT/src:$PYTHONPATH
ipython -i test/closed_loop.py data/par/par4bench/scao_sh_16x16_8pix.py
```
