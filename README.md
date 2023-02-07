# WORK IN PROGRESS: THE CODE MIGHT NOT WORK
# neurograph

## install

python=3.10

### pip
In order to install all dependencies via pip in some virtual env, run:

```bash
./install_pyg_via_pip.sh
pip install -U -r requirements.txt

# install neurograph into env
pip install -e .

```

## data
By default, neurograph expects that your datasets are stored in `datasets` folder e.g. `datasets/cobre`.

# how to use it
Neurograph uses hydra for managing different configurations. See default config in `config/config.py` and `config/config.yaml`

Here we overwrite some default values in config
```bash
python -m neurograph.train \
 hydra.job.name=gat_l2h4_node_concat \
 dataset.data_path=<datasets dir> \
 'model={num_layers: 2, num_heads: 4, prepool_dim: 64, dropout: 0.5, mp_type: node_concate, final_node_dim:8}' \
 train.epochs=20 train.optim_args='{lr: 3.5e-4}'
```

Results will be in `outputs/<hydra.job.name>/<timestamp>` dir

## acronyms
* PyG = pytorch_geometric
* CM = connectivity matrix
* MP = message passing

* subset = train, valid or test part of a whole dataset or of one fold in cross-validation
