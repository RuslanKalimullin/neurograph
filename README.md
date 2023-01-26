# neurograph

Research in GNN applied to multimodal imaging data

## install
```bash
conda env create -f environment.yaml 
conda activate neurograph

# install modified version of BrainGB
# use -e to be able to modify it while working with it
pip install -e BrainGB

# create ipykernel env to use in notebooks
python -m ipykernel install --user --name neurograph 

# install neurograph into your conda env
pip install -e .
```
