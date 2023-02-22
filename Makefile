lint:
	mypy neurograph

test_train_bgbGCN:
	python -m neurograph.train log.wandb_mode=disabled +model=bgbGCN

test_train_bgbGAT:
	python -m neurograph.train log.wandb_mode=disabled +model=bgbGAT

test_train_transformer:
	python -m neurograph.train log.wandb_mode=disabled +model=transformer dataset.data_type=dense

test_train: test_train_bgbGCN test_train_bgbGAT test_train_transformer
