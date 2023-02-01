import argparse
import sys
import torch

# TODO
def main(args):
    device = 'cpu'

    accs, aucs, macros = [], [], []
    models = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(train_ds)), y=train_ds.data.y[train_ds._indices])):
        logging.info(f'Fold {fold_i}')

        train_subset = train_ds[train_idx]
        val_subset = train_ds[val_idx]
        fold_ids[fold_i] = {'train': get_ids(train_subset), 'valid': get_ids(val_subset)}

        # split train into train and valid, create loaders
        train_loader = DataLoader(dataset=train_ds[train_idx], batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset=train_ds[val_idx], batch_size=8, shuffle=False)

        # create model instance
        model = bgb_models.gat.GAT(
            input_dim=116,
            args=config,
            num_nodes=116,
            num_classes=1,
        )
        # set optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # train and eval
        test_micro, test_auc, test_macro = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            optimizer,
            device=device,
            args=config,
        )

        # evaluate again
        test_micro, test_auc, test_macro = evaluate(model, device, val_loader)

        # print valid metrics
        logging.info(f'(Initial Performance Last Epoch) | test_micro={(test_micro * 100):.2f}, '
                     f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')

        # store metrics for the current fold
        accs.append(test_micro)
        aucs.append(test_auc)
        macros.append(test_macro)
        models.append(model)
        del model


# create parser; parse arguments
parser = argparse.ArgumentParser()

# DATASET
parser.add_argument('--dataset_name',
                    type=str,
                    choices=['COBRE'],
                    default="COBRE")
parser.add_argument('--view', type=int, default=1)
parser.add_argument('--node_features', type=str,
                    choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                             'eigenvector', 'eigen_norm'],
                    default='adj')

# MODEL PARAMETERS
parser.add_argument('--model_name', type=str, default='gcn')

# gcn_mp_type choices: weighted_sum, bin_concate, edge_weight_concate, edge_node_concate, node_concate
parser.add_argument('--gcn_mp_type', type=str, default="weighted_sum")

# gat_mp_type choices: attention_weighted, attention_edge_weighted, sum_attention_edge, edge_node_concate, node_concate
parser.add_argument('--gat_mp_type', type=str, default="attention_weighted")

parser.add_argument('--n_GNN_layers', type=int, default=2)
parser.add_argument('--n_MLP_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=360)
parser.add_argument('--gat_hidden_dim', type=int, default=8)
parser.add_argument('--edge_emb_dim', type=int, default=256)
parser.add_argument('--bucket_sz', type=float, default=0.05)

# pooling
parser.add_argument('--pooling', type=str,
                    choices=['sum', 'concat', 'mean'],
                    default='concat')

# TRAINING
parser.add_argument('--enable_nni', action='store_true')
parser.add_argument('--seed', type=int, default=1380)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.5)

#parser.add_argument('--repeat', type=int, default=1)
#parser.add_argument('--k_fold_splits', type=int, default=5)
#parser.add_argument('--diff', type=float, default=0.2)
#parser.add_argument('--mixup', type=int, default=1) #[0, 1]

main(parser.parse_args())
