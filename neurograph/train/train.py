import torch
import logging

from neurograph.config import Config


def build_model_and_loss(cfg: Config):
    pass


def train_one_split(
    model,
    train_loader,
    valid_loader,
    optimizer,
    device,
    cfg: Config,
):
    ''' Pass one split of cross-validation '''

    # set model to train
    model.train()

    # init list with metrics
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs

    # run training for some epochs
    for i in range(epoch_num):
        # total loss for epoch (all batches)
        loss_all = 0

        # get batch
        for data in train_loader:
            # move batch to device
            data = data.to(device)

            # unpack batch
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

            # zero grad
            optimizer.zero_grad()

            # forward
            out = model(x, edge_index, edge_attr, batch)

            # compute loss
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), data.y.float())

            # backward and optimizer step
            loss.backward()
            optimizer.step()

            # update total_loss
            loss_all += loss.item()

        # average by total length of the dataset
        epoch_loss = loss_all / len(train_loader.dataset)

        # compute metrics
        train_micro, train_auc, train_macro = evaluate(model, device, train_loader)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')

        if (i + 1) % test_interval == 0:
            test_micro, test_auc, test_macro = evaluate(model, device, valid_loader)

            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)

            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            logging.info(text)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))

    return accs.mean(), aucs.mean(), macros.mean()


# TODO
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
