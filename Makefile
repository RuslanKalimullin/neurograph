lint:
	mypy neurograph

test_train_bgbGCN:
	python -m neurograph.train log.wandb_mode=disabled +model=bgbGCN +dataset=base_dataset train.device=cpu

test_train_bgbGAT:
	python -m neurograph.train log.wandb_mode=disabled +model=bgbGAT +dataset=base_dataset train.device=cpu

test_train_transformer:
	python -m neurograph.train log.wandb_mode=disabled +model=transformer +dataset=base_dataset  dataset.data_type=dense train.device=cpu

test_dummy_mm2:
	python -m neurograph.train log.wandb_mode=disabled +model=dummy_mm2 +dataset=base_multimodal_dataset dataset.data_type=multimodal_dense_2 train.device=cpu
	
test_train_transfomer_mm2:
	python -m neurograph.train --multirun log.wandb_mode=disabled +dataset=base_multimodal_dataset dataset.name=cobre +model=mm_transformer model.num_layers=1 model.num_heads=2 model.pooling=concat,mean model.hidden_dim=8 model.make_projection=False,True train.epochs=1 train.device=cpu

test_train: test_train_bgbGCN test_train_bgbGAT test_train_transformer test_dummy_mm2 test_train_transfomer_mm2
