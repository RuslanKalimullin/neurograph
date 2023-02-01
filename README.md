# WORK IN PROGRESS: THE CODE MIGHT NOT WORK
# neurograph

## install

### pip
In order to install all dependencies via pip in some virtual env, run:

```bash
./install_pyg_via_pip.sh
pip install -U -r requirements.txt

# ipykernel, pytest, mypy etc.
pip install -U -r requirements.dev.txt

# install neurograph into env
pip install -e .

```

## data
By default, neurograph expects that your datasets are stored in `datasets` folder e.g. `datasets/cobre_fmri`.

## acronyms
* PyG = pytorch_geometric
* CM = connectivity matrix
