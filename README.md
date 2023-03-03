# WORK IN PROGRESS: THE CODE MIGHT NOT WORK
# neurograph

## install
### Basic requirements
python=3.10, torch=1.12.1, cuda=11.3

### pip
In order to install all dependencies via pip in some virtual env, run:

```bash
# required for constructing 2-complexes for CWN
# don't install it, if you're training just GNN
./graph-tool_install.sh  # via conda

# install Pytorch Geometric
./install_pyg.sh cu113 # or ./pyg_cpu.sh cpu

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

### COBRE
Cobre dataset is used by default


Run gridsearch for bgbGAT, bgbGCN:

```bash
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

Run gridsearch for vanilla transformers:
```bash
python -m neurograph.train --multirun dataset.data_type=dense +model=transformer8,transformer16,transformer32,transformer64,transformer116 model.num_layers=1,2 model.num_heads=1,2,4 model.pooling=concat,mean dataset.feature_type=conn_profile,timeseries train.scheduler=null train.device="cuda:0" train.epochs=100
```

### PPMI
Since PPMI has only DTI data we need to change some default params. Also, DTI data usually needs some normalization since connectivity matrices contain a number of detected tract between different ROI
```bash
python -m <other params> dataset.name=ppmi dataset.experiment_type=dti dataset.normalize=log
```

Results will be logged into wandb

## Docker
```bash
# build an image
docker build -t neurograph .

# run it as a container and do your stuff inside
docker run -it --rm --network host --gpus=0,1 -v $(pwd):/app neurograph /bin/bash

# or run a particular gridsearch e.g.
docker run --rm --network host --gpus=0,1 -v $(pwd):/app --env WANDB_API_KEY=<YOUR_WANDB_API_KEY> neurograph bash -c 'python -m neurograph.train --multirun log.wandb_project=mri_docker_test +model=transformer8,transformer16 ++model.num_layers=1 model.num_heads=1,8 dataset.data_type=dense train.scheduler=null'
```

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
