import numpy as np
import torch


def eval_multi_clf(model, dev_data_loader, device, logger=None, T=1.0):
    model.eval()
    labels, all_logits = [], []
    with torch.no_grad():
        for batch in dev_data_loader:
            batch_data = [i.to(device) for i in batch]
            logits, loss_value = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                       token_type_ids=batch_data[2], labels=batch_data[3])

            logits = logits.to("cpu").numpy()

            new_labels = batch_data[3].to("cpu").numpy()[:, 1:]

            labels.append(new_labels)
            all_logits.append(logits)

    labels = np.vstack(labels)
    all_logits = np.vstack(all_logits)

    count = 0
    total_len = len(labels)
    for dd_label, dd_pred in zip(labels, all_logits):

        dd_pred = np.array([round(i) for i in dd_pred])
        dd_label = np.array([round(i) for i in dd_label])
        if (dd_pred == dd_label).all():
            count += 1
    if logger is not None:
        logger.info('right: {}\ttotal: {}\tM-tree codes acc: {}'.format(count, total_len, count / total_len))

    model.train()
    return count / total_len
