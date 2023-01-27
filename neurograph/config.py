import pathlib

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent/ 'datasets'

if not DATA_PATH.exists():
    DATA_PATH.mkdir()
