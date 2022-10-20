import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from .MwpDataset import MwpDataSet, process_mwp_V1, process_mwp_V1_plus
from .Models import MwpBertModel, MwpBertModel_CLS
from .verification_labels_plus import re_construct_expression_from_codes, build_expression_by_grous, verification
from .Utils import process_one_mawps_for_test


def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    logging.info(f"loading model from {str(model_path)} .")
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states['state_dict']
    if isinstance(model, torch.nn.DataParallel):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


def check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
    true_ans = raw_mwp['new_ans']
    T_number_map = raw_mwp['T_number_map']
    T_question = raw_mwp['T_question']

    pre_num_codes = get_codes_from_output(labels_pos, all_logits, T_question, id2label_or_value)

    symbol2num = re_construct_expression_from_codes(pre_num_codes)
    final_expression = build_expression_by_grous(symbol2num)

    if final_expression == '':
        raw_mwp['pre_final_expression'] = 'Failed'
        return False
    else:
        raw_mwp['pre_final_expression'] = final_expression
        if verification(final_expression, T_number_map, true_ans):
            return True
        else:
            return False


def check_codes_acc(raw_mwp, labels, all_logits):
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

    raw_mwp['codes_pre_results'] = right_wrong_vector
    if count == total_len:
        return True
    else:
        return False


def get_codes_from_output(labels, all_logits, T_question, id2label_or_value):
    pre_num_codes = {}

    T_question_list = list(T_question)
    for dd_label, dd_pred in zip(labels, all_logits):

        dd_label = np.array([round(i) for i in dd_label])
        num_index = dd_label[0]
        num_char = T_question_list[num_index]

        if num_char not in pre_num_codes.keys():
            pre_num_codes[num_char] = []

        dd_pred = np.array([round(i) for i in dd_pred])
        for idid, vvv in enumerate(dd_pred):
            if vvv > 0:
                codeee = id2label_or_value[str(idid)]
                if codeee != 'None':
                    for i in range(int(vvv)):
                        pre_num_codes[num_char].append(codeee)
    return pre_num_codes


def test_for_mwp_BERT_slover(args, false_file_name: str = None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path_for_test)

    print(">>>>>loading label2id file...")
    label2id_or_value = {}
    id2label_or_value = {}

    if isinstance(args.label2id_path, list):
        label2id_list = args.label2id_path
    elif isinstance(args.label2id_path, str):
        with open(args.label2id_path, 'r', encoding='utf-8') as fr:
            label2id_list = json.load(fr)
    else:
        print('file parameter wrong !!! \n exit !!!')
        sys.exit()

    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde
        id2label_or_value[str(inde)] = label

    print(">>>>>define model...")
    model_fc_path = os.path.join(args.pretrain_model_path_for_test, 'fc_weight.bin')
    if not args.use_cls:
        clf_model = MwpBertModel(bert_path_or_config=args.pretrain_model_path_for_test, num_labels=args.num_labels,
                                 fc_path=model_fc_path, multi_fc=args.multi_fc, train_loss=args.train_loss,
                                 fc_hidden_size=args.fc_hidden_size)
    else:
        clf_model = MwpBertModel_CLS(bert_path_or_config=args.pretrain_model_path_for_test, num_labels=args.num_labels,
                                     fc_path=model_fc_path, multi_fc=args.multi_fc, train_loss=args.train_loss,
                                     fc_hidden_size=args.fc_hidden_size)
    clf_model.to(args.device)

    if args.use_multi_gpu:
        clf_model = torch.nn.DataParallel(clf_model)

    if args.test_data_path:
        print(">>>>>loading test data...")

        if isinstance(args.test_data_path, list):
            test_mwps = args.test_data_path
        elif isinstance(args.test_data_path, str):
            with open(args.test_data_path, 'r', encoding='utf-8') as fr:
                test_mwps = json.load(fr)
        else:
            print('file parameter wrong !!! \n exit !!!')
            sys.exit()

        right_codes_count = 0
        right_ans_count = 0
        false_mwp = []
        for idd, raw_mwp in enumerate(test_mwps):
            processed_mwp = process_one_mawps_for_test(raw_mwp, label2id_or_value, args.test_dev_max_len, True,
                                                       tokenizer)

            if processed_mwp is not None:
                batch = list(zip(*processed_mwp))
                batch = [torch.tensor(batch[0]).long(), torch.tensor(batch[1]).long(), torch.tensor(batch[2]).long(),
                         torch.tensor(batch[3]).long()]

                clf_model.eval()
                labels, all_logits = [], []
                labels_pos = []

                with torch.no_grad():
                    batch_data = [i.to(args.device) for i in batch]
                    logits, loss_value = clf_model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], labels=batch_data[3])
                    logits = logits.to("cpu").numpy()
                    new_labels = batch_data[3].to("cpu").numpy()[:, 1:]
                    labels.append(new_labels)
                    labels_pos.append(batch_data[3].to("cpu").numpy())
                    all_logits.append(logits)

                labels = np.vstack(labels)
                all_logits = np.vstack(all_logits)
                labels_pos = np.vstack(labels_pos)

                if check_codes_acc(raw_mwp, labels, all_logits):
                    right_codes_count += 1
                if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
                    right_ans_count += 1
                else:
                    false_mwp.append(raw_mwp)

            else:
                print('数据处理失败！！！')
                continue

        total_acc = right_ans_count / len(test_mwps)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), total_acc))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if false_file_name is not None:
            print('Write test false data into: {}'.format(os.path.join('./false_data', false_file_name)))
            with open(os.path.join('./false_data/', false_file_name), 'w', encoding='utf-8') as false_out:
                false_out.write(json.dumps(false_mwp, ensure_ascii=False, indent=2))

    else:
        print("not provide test file path")
        sys.exit()
