# WORK IN PROGRESS: THE CODE MIGHT NOT WORK
# neurograph

## install

python=3.10

### pip
In order to install all dependencies via pip in some virtual env, run:

```bash
# required for constructing 2-complexes for CWN
# don't install it, if you're training just GNN
./graph-tool_install.sh  # via conda

# install Pytorch Geometric
./pyg_cpu.sh  # or ./pyg_cuda.sh

# install other requirements
pip install -U -r requirements.txt

# install `neurograph` into env
pip install -e .

# CWN: install `cwn` fork (maybe to another dir)
git clone https://github.com/gennadylaptev/cwn.git
cd cwn
pip install -e .
```

## data
By default, neurograph expects that your datasets are stored in `datasets` folder e.g. `datasets/cobre`.

# how to use it
Neurograph uses `hydra` for managing different configurations. See default config in `config/config.py` and `config/config.yaml`

```bash
# Run bgbGAT, bgbGCN
!python -m neurograph.train --multirun \
  dataset.data_path='<path_to_data>' \
  +model=bgbGAT  # bgbGCN \
  model.num_layers=1,2 \
  model.num_heads=1,2,4 \
  model.hidden_dim=8,12,16 \
  dataset.pt_thr=0.25,0.5,0.75,null \
  train.epochs=20 \
  train.scheduler=null
```

Results will be logged into wandb

## acronyms
* PyG = pytorch_geometric
* CM = connectivity matrix
* MP = message passing

* subset = train, valid or test part of a whole dataset or of one fold in cross-validation

## misc
```bash
jupyter nbextension enable --py widgetsnbextension
```
if IPython doesn't work in jupyter lab
