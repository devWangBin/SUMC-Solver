import json
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.MwpDataset import MwpDataSet
from src.Models import MwpBertModel, MwpBertModel_CLS
from src.Evaluation import eval_multi_clf


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    if args.train_type != 'together':

        logger.info("get train data loader...")

        train_data_set = MwpDataSet(cached_path=args.cached_data_path, file_path=args.train_data_path,
                                    tokenizer=tokenizer, label2id_data_path=args.label2id_path,
                                    max_len=args.train_max_len, has_label=True,
                                    use_new_token_type_id=args.use_new_token_type_id)

        if args.train_type == 'one-by-one-in-same-batch':
            train_data_loader = DataLoader(dataset=train_data_set, batch_size=args.batch_size,
                                           sampler=SequentialSampler(train_data_set))
        elif args.train_type == 'one-by-one-random':
            train_data_loader = DataLoader(dataset=train_data_set, batch_size=args.batch_size,
                                           sampler=RandomSampler(train_data_set))
        else:
            print('args.train_type wrong!!!')
            sys.exit()

        logger.info("get dev data loader...")
        dev_data_set = MwpDataSet(cached_path=args.cached_data_path, file_path=args.dev_data_path,
                                  tokenizer=tokenizer, has_label=True,
                                  label2id_data_path=args.label2id_path, max_len=args.test_dev_max_len,
                                  use_new_token_type_id=args.use_new_token_type_id)
        dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=args.batch_size,
                                     sampler=SequentialSampler(dev_data_set))

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
    if not args.use_cls:
        model = MwpBertModel(bert_path_or_config=args.pretrain_model_path, num_labels=args.num_labels,
                             fc_path=args.fc_path, multi_fc=args.multi_fc, train_loss=args.train_loss,
                             fc_hidden_size=args.fc_hidden_size)
    else:
        model = MwpBertModel_CLS(bert_path_or_config=args.pretrain_model_path, num_labels=args.num_labels,
                                 fc_path=args.fc_path, multi_fc=args.multi_fc, train_loss=args.train_loss,
                                 fc_hidden_size=args.fc_hidden_size)

    if args.use_multi_gpu:
        print('*********************************************************')
        print('**************** multiple GPU training ******************')
        print(torch.cuda.device_count())
        print('*********************************************************')
        model.to(args.device)
        model = torch.nn.DataParallel(model)

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
            "weight_decay": 0.01,
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

    global_steps = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            global_steps += 1
            batch_data = [i.to(args.device) for i in batch]

            logits, loss = model(input_ids=batch_data[0], attention_mask=batch_data[1], token_type_ids=batch_data[2],
                                 labels=batch_data[3])

            if step <= 1:
                print(batch_data[0].shape)
                print(batch_data[1].shape)
                print(batch_data[2].shape)
                print(batch_data[3].shape)
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

                acc = eval_multi_clf(
                    logger=logger,
                    model=model,
                    dev_data_loader=dev_data_loader,
                    device=args.device)

                if acc > best_acc:
                    logger.info('save best model to {}'.format(best_model_dir))
                    best_acc = acc
                    model.save(save_dir=best_model_dir)
                model.save(save_dir=latest_model_dir)

        if args.re_process_train_data and args.train_type == 'one-by-one-in-same-batch':
            logger.info("Repeatedly get train data loader...")
            train_data_set = MwpDataSet(cached_path=args.cached_data_path, file_path=args.train_data_path,
                                        tokenizer=tokenizer, use_cache=False,
                                        label2id_data_path=args.label2id_path, max_len=args.train_max_len,
                                        use_new_token_type_id=args.use_new_token_type_id, has_label=True)

            train_data_loader = DataLoader(dataset=train_data_set, batch_size=args.batch_size,
                                           sampler=SequentialSampler(train_data_set))
