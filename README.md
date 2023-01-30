# neurograph

Research in GNN applied to multimodal imaging data

## install
### conda
```bash
conda env create -f environment.yaml 
conda activate neurograph

# install modified version of BrainGB; clone it somewhere
git clone https://github.com/gennadylaptev/BrainGB.git
# use -e to be able to modify it while working with it
pip install -e BrainGB

# create ipykernel env to use in notebooks
python -m ipykernel install --user --name neurograph 

# install neurograph into your conda env
pip install -e .
```

### pip
In order to install all dependencies via pip in some virtual env, run:
```bash
./install_pyg_via_pip.sh
pip install -U -r requirements.txt
# ipykernel, pytest, mypy etc.
pip install -U -r requirements.dev.txt
```

## data
By default, neurograph expects that your datasets are stored in `datasets` folder e.g. `datasets/cobre_fmri`.

## acronyms
* CM = connectivity matrix
