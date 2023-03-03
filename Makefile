lint:
	mypy neurograph

test_train_bgbGCN:
	python -m neurograph.train log.wandb_mode=disabled +model=bgbGCN +dataset=base_dataset

test_train_bgbGAT:
	python -m neurograph.train log.wandb_mode=disabled +model=bgbGAT +dataset=base_dataset

test_train_transformer:
	python -m neurograph.train log.wandb_mode=disabled +model=transformer +dataset=base_dataset  dataset.data_type=dense

test_dummy_mm2:
	python -m neurograph.train log.wandb_mode=disabled +model=dummy_mm2 +dataset=base_multimodal_dataset dataset.data_type=multimodal_dense_2

test_train: test_train_bgbGCN test_train_bgbGAT test_train_transformer test_dummy_mm2
