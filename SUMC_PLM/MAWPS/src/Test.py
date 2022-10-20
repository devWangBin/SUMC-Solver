import json
import logging
import os
import sys
import json
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

from .MwpDataset import MwpDataSet, process_mwp_V1, process_mwp_V1_plus
from .Models import MwpBertModel, MwpBertModel_CLS
from .Evaluation import eval_multi_clf
from tqdm import tqdm
import numpy as np


def test_for_mwp_slover_old(conf, CLS: bool = False, false_file_name: str = 'test_false_mwp.json'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path_for_test)

    print(">>>>>loading label2id file...")
    label2id_or_value = {}
    with open(conf.label2id_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde

    print(">>>>>define model...")
    model_fc_path = os.path.join(conf.pretrain_model_path_for_test, 'fc_weight.bin')
    if not CLS:
        clf_model = MwpBertModel(bert_path_or_config=conf.pretrain_model_path_for_test, num_labels=conf.num_labels,
                                 fc_path=model_fc_path, multi_fc=conf.multi_fc)
    else:
        clf_model = MwpBertModel_CLS(bert_path_or_config=conf.pretrain_model_path_for_test, num_labels=conf.num_labels,
                                     fc_path=model_fc_path, multi_fc=conf.multi_fc)
    clf_model.to(conf.device)

    if conf.test_data_path and os.path.exists(conf.test_data_path):
        print(">>>>>loading test data...")
        with open(conf.test_data_path, 'r', encoding='utf-8')as ffs:
            test_mwps = json.load(ffs)

        right_count = 0
        false_mwp = []
        pbar = tqdm(test_mwps)
        for idd, raw_mwp in enumerate(pbar):

            result_sens, result_labels, _ = process_mwp_V1(raw_mwp, label2id_or_value)
            if result_sens is not None:

                res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=result_sens,
                                                  padding='max_length',
                                                  add_special_tokens=True,
                                                  max_length=conf.test_dev_max_len,
                                                  truncation=True,
                                                  return_tensors='pt',
                                                  )
                batch = [res['input_ids'], res['attention_mask'], res['token_type_ids'],
                         torch.tensor(result_labels).long()]

                clf_model.eval()
                labels, all_logits = [], []

                with torch.no_grad():
                    batch_data = [i.to(conf.device) for i in batch]
                    logits, loss_value = clf_model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], labels=batch_data[3])

                    logits = logits.to("cpu").numpy()
                    new_labels = batch_data[3].to("cpu").numpy()[:, 1:]

                    labels.append(new_labels)
                    all_logits.append(logits)

                labels = np.vstack(labels)
                all_logits = np.vstack(all_logits)

                count = 0
                total_len = len(labels)
                right_wrong_vector = []
                for dd_label, dd_pred in zip(labels, all_logits):

                    dd_pred = np.array([round(i) for i in dd_pred])
                    dd_label = np.array([round(i) for i in dd_label])
                    if (dd_pred == dd_label).all():
                        count += 1
                        right_wrong_vector.append(1)
                    else:
                        print('***************************************************')
                        print(dd_pred)
                        print(dd_label)
                        print('***************************************************')
                        right_wrong_vector.append(0)

                if count == total_len:
                    right_count += 1
                else:
                    raw_mwp['labels_pre_results'] = right_wrong_vector
                    false_mwp.append(raw_mwp)

            else:
                print('数据处理失败！！！')
                continue

        total_acc = right_count / len(test_mwps)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_count, len(test_mwps), total_acc))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        print('Write test false data into: {}'.format(os.path.join('./Dataset', false_file_name)))
        with open(os.path.join('./Dataset/', false_file_name), 'w', encoding='utf-8') as false_out:
            false_out.write(json.dumps(false_mwp, ensure_ascii=False, indent=2))


    else:
        print("not provide test file path")

        sys.exit()


def test_for_mwp_slover(conf, CLS: bool = False, false_file_name: str = 'test_false_mwp.json'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path_for_test)

    print(">>>>>loading label2id file...")
    label2id_or_value = {}
    with open(conf.label2id_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde

    print(">>>>>define model...")
    model_fc_path = os.path.join(conf.pretrain_model_path_for_test, 'fc_weight.bin')
    if not CLS:
        clf_model = MwpBertModel(bert_path_or_config=conf.pretrain_model_path_for_test, num_labels=conf.num_labels,
                                 fc_path=model_fc_path, multi_fc=conf.multi_fc)
    else:
        clf_model = MwpBertModel_CLS(bert_path_or_config=conf.pretrain_model_path_for_test, num_labels=conf.num_labels,
                                     fc_path=model_fc_path, multi_fc=conf.multi_fc)
    clf_model.to(conf.device)
    if conf.use_multi_gpu:
        clf_model = torch.nn.DataParallel(clf_model)

    if conf.test_data_path and os.path.exists(conf.test_data_path):
        print(">>>>>loading test data...")
        with open(conf.test_data_path, 'r', encoding='utf-8')as ffs:
            test_mwps = json.load(ffs)

        right_count = 0
        false_mwp = []
        pbar = tqdm(test_mwps)
        for idd, raw_mwp in enumerate(pbar):

            result_sens, result_labels, num_pos = process_mwp_V1(raw_mwp, label2id_or_value)
            if result_sens is not None:

                res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=result_sens,
                                                  padding='max_length',
                                                  add_special_tokens=True,
                                                  max_length=conf.test_dev_max_len,
                                                  truncation=True,
                                                  return_tensors='pt',
                                                  )
                if conf.use_new_token_type_id:
                    new_token_type_ids = []
                    for num_ps in num_pos:
                        tem = [0] * conf.test_dev_max_len
                        for pp in num_ps:
                            tem[pp + 1] = 1
                        new_token_type_ids.append(tem)
                    new_token_type_ids = torch.tensor(new_token_type_ids).long()
                    res['token_type_ids'] = new_token_type_ids

                batch = [res['input_ids'], res['attention_mask'], res['token_type_ids'],
                         torch.tensor(result_labels).long()]

                clf_model.eval()
                labels, all_logits = [], []

                with torch.no_grad():
                    batch_data = [i.to(conf.device) for i in batch]
                    logits, loss_value = clf_model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], labels=batch_data[3])

                    logits = logits.to("cpu").numpy()
                    new_labels = batch_data[3].to("cpu").numpy()[:, 1:]

                    labels.append(new_labels)
                    all_logits.append(logits)

                labels = np.vstack(labels)
                all_logits = np.vstack(all_logits)

                count = 0
                total_len = len(labels)
                right_wrong_vector = []
                for dd_label, dd_pred in zip(labels, all_logits):

                    dd_pred = np.array([round(i) for i in dd_pred])
                    dd_label = np.array([round(i) for i in dd_label])
                    if (dd_pred == dd_label).all():
                        count += 1
                        right_wrong_vector.append(1)
                    else:

                        right_wrong_vector.append(0)

                if count == total_len:
                    right_count += 1
                else:
                    raw_mwp['labels_pre_results'] = right_wrong_vector
                    false_mwp.append(raw_mwp)

            else:
                print('数据处理失败！！！')
                continue

        total_acc = right_count / len(test_mwps)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_count, len(test_mwps), total_acc))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        print('Write test false data into: {}'.format(os.path.join('./Dataset', false_file_name)))
        with open(os.path.join('./Dataset/', false_file_name), 'w', encoding='utf-8') as false_out:
            false_out.write(json.dumps(false_mwp, ensure_ascii=False, indent=2))


    else:
        print("not provide test file path")

        sys.exit()


def test_for_mwp_slover2(conf, CLS: bool = False, false_file_name: str = None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path_for_test)

    print(">>>>>loading label2id file...")
    label2id_or_value = {}
    with open(conf.label2id_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde

    print(">>>>>define model...")
    model_fc_path = os.path.join(conf.pretrain_model_path_for_test, 'fc_weight.bin')
    if not CLS:
        clf_model = MwpBertModel(bert_path_or_config=conf.pretrain_model_path_for_test, num_labels=conf.num_labels,
                                 fc_path=model_fc_path, multi_fc=conf.multi_fc, train_loss=conf.train_loss,
                                 fc_hidden_size=conf.fc_hidden_size)
    else:
        clf_model = MwpBertModel_CLS(bert_path_or_config=conf.pretrain_model_path_for_test, num_labels=conf.num_labels,
                                     fc_path=model_fc_path, multi_fc=conf.multi_fc, train_loss=conf.train_loss,
                                     fc_hidden_size=conf.fc_hidden_size)
    clf_model.to(conf.device)

    if conf.use_multi_gpu:
        clf_model = torch.nn.DataParallel(clf_model)

    total_label_wrong_count = [0] * 144
    total_label_wrong_count = np.array(total_label_wrong_count, dtype=float)

    if conf.test_data_path and os.path.exists(conf.test_data_path):
        print(">>>>>loading test data...")
        with open(conf.test_data_path, 'r', encoding='utf-8')as ffs:
            test_mwps = json.load(ffs)

        right_count = 0
        false_mwp = []
        pbar = tqdm(test_mwps)
        for idd, raw_mwp in enumerate(pbar):

            result_sens, result_labels, num_pos, all_num_label_all = process_mwp_V1_plus(raw_mwp, label2id_or_value)
            if result_sens is not None:

                res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=result_sens,
                                                  padding='max_length',
                                                  add_special_tokens=True,
                                                  max_length=conf.test_dev_max_len,
                                                  truncation=True,
                                                  return_tensors='pt',
                                                  )
                if conf.use_new_token_type_id:
                    new_token_type_ids = []
                    for num_ps in num_pos:
                        tem = [0] * conf.test_dev_max_len
                        for pp in num_ps:
                            tem[pp + 1] = 1
                        new_token_type_ids.append(tem)
                    new_token_type_ids = torch.tensor(new_token_type_ids).long()
                    res['token_type_ids'] = new_token_type_ids

                batch = [res['input_ids'], res['attention_mask'], res['token_type_ids'],
                         torch.tensor(result_labels).long()]

                clf_model.eval()
                labels, all_logits = [], []

                with torch.no_grad():
                    batch_data = [i.to(conf.device) for i in batch]
                    logits, loss_value = clf_model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], labels=batch_data[3])

                    logits = logits.to("cpu").numpy()
                    new_labels = batch_data[3].to("cpu").numpy()[:, 1:]

                    labels.append(new_labels)
                    all_logits.append(logits)

                labels = np.vstack(labels)
                all_logits = np.vstack(all_logits)

                count = 0
                total_len = len(labels)
                right_wrong_vector = []

                assert len(labels) == len(all_logits) == len(all_num_label_all)

                result_details = []

                for dd_label_raw, dd_pred_raw, txtx in zip(labels, all_logits, all_num_label_all):

                    dd_pred = np.array([round(i) for i in dd_pred_raw])
                    dd_label = np.array([round(i) for i in dd_label_raw])

                    dd_pred_raw = ' '.join([str(round(i, 3)) for i in dd_pred_raw])
                    dd_label_raw = ' '.join([str(round(i, 3)) for i in dd_label_raw])

                    wasawas = ' '.join([str(int(i)) for i in dd_pred.tolist()])

                    if (dd_pred == dd_label).all():
                        count += 1
                        right_wrong_vector.append(1)
                    else:

                        right_wrong_vector.append(0)
                        total_label_wrong_count = total_label_wrong_count + dd_label

                    txtx.append(dd_pred_raw)
                    txtx.append(wasawas)
                    txtx.append(dd_label_raw)
                    result_details.append(txtx)

                if count == total_len:
                    right_count += 1
                else:
                    raw_mwp['labels_pre_results'] = right_wrong_vector
                    raw_mwp['result_details'] = result_details
                    false_mwp.append(raw_mwp)

            else:
                print('数据处理失败！！！')
                continue

        total_acc = right_count / len(test_mwps)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_count, len(test_mwps), total_acc))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if false_file_name is not None:
            print('Write test false data into: {}'.format(os.path.join('./false_data', false_file_name)))
            with open(os.path.join('./false_data/', false_file_name), 'w', encoding='utf-8') as false_out:
                false_out.write(json.dumps(false_mwp, ensure_ascii=False, indent=2))

            with open('./false_data/total_label_wrong_count_' + false_file_name + '.json', 'w',
                      encoding='utf-8') as false_out2:
                false_out2.write(json.dumps(list(total_label_wrong_count), ensure_ascii=False, indent=2))

    else:
        print("not provide test file path")

        sys.exit()
