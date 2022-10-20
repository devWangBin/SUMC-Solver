import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from .Utils import process_dataset, MWPDatasetLoader
from .evaluation import eval_multi_clf_rnn
from .model import RNNModel, RNNModel_ATT
import torch.nn as nn


def save_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


def train_fast(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('device: '.format(args.device))

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    latest_model_dir = os.path.join(args.output_dir, "latest_model")
    os.makedirs(latest_model_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("get train data loader...")
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    args.vocab_size = tokenizer.vocab_size

    if args.train_type != 'together':

        logger.info("get train data loader...")
        examplesss_train = process_dataset(file=args.train_data_path, label2id_data_path=args.label2id_path,
                                           max_len=args.train_max_len, lower=True)

        if args.train_type == 'one-by-one-in-same-batch':

            train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=False,
                                                 tokenizer=tokenizer, seed=72, sort=False)
        elif args.train_type == 'one-by-one-random':
            train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=True,
                                                 tokenizer=tokenizer, seed=72, sort=False)
        else:
            print('args.train_type wrong!!!')
            sys.exit()

        logger.info("get dev data loader...")
        examplesss_test = process_dataset(file=args.dev_data_path, label2id_data_path=args.label2id_path,
                                          max_len=args.test_dev_max_len, lower=True)

        dev_data_loader = MWPDatasetLoader(data=examplesss_test, batch_size=args.batch_size, shuffle=False,
                                           tokenizer=tokenizer, seed=72, sort=False)
    else:
        print('args.train_type == together not yet!!!')
        sys.exit()

    total_steps = int(len(train_data_loader) * args.num_epochs)
    steps_per_epoch = len(train_data_loader)

    if args.warmup < 1:
        warmup_steps = int(total_steps * args.warmup)
    else:
        warmup_steps = int(args.warmup)

    logger.info("define model...")
    if args.use_attention:
        MModel = RNNModel_ATT
    else:
        MModel = RNNModel

    model = MModel(vocab_size=args.vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                   num_layer=args.rnn_layer, fc_size=args.fc_size, rnn_cell=args.rnn_cell,
                   num_labels=args.num_labels, fc_path=args.fc_path, multi_fc=args.multi_fc, drop_p=args.dropout)

    if args.train_model_path is not None:
        logger.info('loading model from path......')
        model = load_model(model, args.train_model_path)

    model.to(args.device)

    model.zero_grad()
    model.train()

    logger.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    logger.info("===========================train setting parameters=========================")
    for n, p in paras.items():
        logger.info("{}-{}".format(n, str(p.shape)))
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        }, {
            "params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    best_acc = -1
    logger.info("\n>>>>>>>>>>>>>>>>>>>start train......")

    args.log_steps = int(steps_per_epoch / 10)

    log_best_acc = os.path.join(best_model_dir, 'best-codes-acc.txt')
    ff_best = open(log_best_acc, 'a', encoding='utf-8')
    log_test_acc = os.path.join(latest_model_dir, 'latest-codes-acc.txt')
    ff_test = open(log_test_acc, 'a', encoding='utf-8')

    global_steps = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            global_steps += 1
            batch_data = [i.to(args.device) for i in batch]

            logits, loss = model(inputs_ids=batch_data[0], input_mask=batch_data[1], labels=batch_data[3])

            if step == epoch:
                logger.info(str(batch_data[0].shape))
                logger.info(str(batch_data[1].shape))
                logger.info(str(batch_data[2].shape))
                logger.info(str(batch_data[3].shape))
                print('input_ids:', batch_data[0][0].cpu().numpy().tolist())
                print("=" * 20)
                print('input_ids:', tokenizer.decode(batch_data[0][0].cpu().numpy().tolist()))
                print("=" * 20)
                print('attention_mask:', batch_data[1][0].cpu().numpy().tolist())
                print("=" * 20)
                print('token_type_ids:', batch_data[2][0].cpu().numpy().tolist())
                print("=" * 20)
                print('label:{}'.format(batch_data[3][0].cpu().numpy().tolist()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if global_steps % args.log_steps == 0:
                logger.info("epoch:{},\tsteps{}/{},\tloss:{}".format(epoch, step, steps_per_epoch, loss.item()))

            if global_steps % steps_per_epoch == 0:

                logger.info("start evaluate...")
                logger.info("=" * 20 + "dev data set" + "=" * 20)

                acc = eval_multi_clf_rnn(
                    logger=logger,
                    model=model,
                    dev_data_loader=dev_data_loader,
                    device=args.device)

                if isinstance(model, nn.DataParallel):
                    model_stat_dict = model.module.state_dict()
                else:
                    model_stat_dict = model.state_dict()
                state = {'epoch': epoch, 'arch': args.model_name, 'state_dict': model_stat_dict}

                if acc > best_acc:
                    logger.info('save best model to {}'.format(best_model_dir))
                    best_acc = acc
                    model_path = os.path.join(best_model_dir, 'best-model.bin')
                    torch.save(state, str(model_path))
                    ff_best.write(str(best_acc) + '\n')
                    ff_best.flush()

                model_path = os.path.join(latest_model_dir, 'last-model.bin')
                torch.save(state, str(model_path))
                ff_test.write(str(acc) + '\n')
                ff_test.flush()

        print('train shuffle doing reste......')
        print('train shuffle doing reste......')
        train_data_loader.reset(doshuffle=True)

    ff_best.close()
    ff_test.close()


class Getoutofloop(Exception):
    pass
