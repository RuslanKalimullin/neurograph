import argparse
import torch
import logging
from neurograph.data.datasets import CobreDataset


def train_split(
    model,
    train_loader,
    test_loader,
    optimizer,
    device,
    args,
):
    # set model to train
    model.train()

    # init list
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs

    for i in range(epoch_num):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

            # zero grads
            optimizer.zero_grad()

            # forward
            out = model(x, edge_index, edge_attr, batch)

            # compute loss
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), data.y.float())

            # backward and optimizer step
            loss.backward()
            optimizer.step()

            loss_all += loss.item()

        epoch_loss = loss_all / len(train_loader.dataset)

        train_micro, train_auc, train_macro = evaluate(model, device, train_loader)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')

        if (i + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)

            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))

    return accs.mean(), aucs.mean(), macros.mean()


@torch.no_grad()
def evaluate(model, device, loader, thr=0.5,):
    # compute metrics on valid data

    model.eval()
    preds, trues, preds_prob = [], [], []

    for data in loader:
        data = data.to(device)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        c = model(x, edge_index, edge_attr, batch)

        # label preds
        preds += (torch.sigmoid(c) > thr).long().detach().cpu().tolist()
        preds_prob += (torch.sigmoid(c)).detach().cpu().tolist()
        trues += data.y.detach().long().cpu().tolist()

    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(train_auc):
        train_auc = 0.5

    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    return train_micro, train_auc, train_macro


def train(args: argparse.Namespace):
    ''' run end-to-end training '''

    # load dataset
    assert args.dataset_name in {'COBRE'}


