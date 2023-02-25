# pass `cu113` or `cpu` as the first arg
DEVICE=$1
pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/${DEVICE}
pip install torch-geometric==2.0.3 torch-scatter torch-sparse  torch-cluster torch-spline-conv  -f https://data.pyg.org/whl/torch-1.13.0+{DEVICE}.html
