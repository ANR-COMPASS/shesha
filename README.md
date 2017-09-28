Installation de COMPASS via conda

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH=/home/sevin/miniconda3/bin:$PATH
conda install -c compass compass -y
```

Installation de SHESHA package for COMPASS

```sh
cd
git clone https://github.com/ANR-COMPASS/shesha.git
export SHESHA_ROOT=$HOME/shesha
export PYTHONPATH=$SHESHA_ROOT/src:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
cd $SHESHA_ROOT
ipython -i test/closed_loop.py data/par/par4bench/scao_sh_16x16_8pix.py
```
