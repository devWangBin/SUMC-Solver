import json
import os
import sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


class MwpDataSet(Dataset):
    def __init__(self, cached_path: str, file_path: str, tokenizer: BertTokenizer, label2id_data_path: str,
                 max_len: int, has_label: bool, use_new_replace: bool):
        self.tokenizer = tokenizer

        label2id_or_value = {}
        with open(label2id_data_path, 'r', encoding='utf-8')as ff:
            label2id_list = json.load(ff)
        for inde, label in enumerate(label2id_list):
            label2id_or_value[label] = inde

        self.label2id_or_value = label2id_or_value
        self.num_class = len(label2id_or_value)
        self.max_len = max_len
        self.has_label = has_label

        type_name = file_path.split('/')[-1].split('.')[0]

        cached_file = os.path.join(cached_path, '{}_{}'.format(type_name, max_len))
        print('cached_file: ', cached_file)

        if use_new_replace:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> use_new_replace !!! >>>>>>>>>>>>>>>>>>>>>>')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> use_new_replace !!! >>>>>>>>>>>>>>>>>>>>>>')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> use_new_replace !!! >>>>>>>>>>>>>>>>>>>>>>')

        if os.path.exists(cached_file):
            print('Loading data from existed cached file %s', cached_file)
            res = torch.load(cached_file)
            self.input_ids, self.attention_mask, self.token_type_ids = res['input_ids'], res['attention_mask'], res[
                'token_type_ids']
            if has_label:
                self.labels = res['labels']
            print('>>>>>>>>>>>>>Dataset size: {}'.format(self.__len__()))
        else:

            sens = []
            labels = []
            num_pos = []
            count_pass = 0
            with open(file_path, 'r', encoding='utf-8') as fr:
                dataset = json.load(fr)
            pbar = tqdm(dataset)
            for idd, raw_mwp in enumerate(pbar):

                if not use_new_replace:
                    result_sens, result_labels, num_postions, result_sens_lens = process_mwp_V1(raw_mwp,
                                                                                                self.label2id_or_value,
                                                                                                self.max_len)
                else:
                    result_sens, result_labels, num_postions, result_sens_lens = process_mwp_V2(raw_mwp,
                                                                                                self.label2id_or_value,
                                                                                                self.max_len)

                if result_sens is not None:
                    sens.extend(result_sens)
                    labels.extend(result_labels)
                    num_pos.extend(num_postions)
                else:
                    count_pass += 1
                    continue

            assert len(sens) == len(labels) == len(num_pos)

            print('total pass {}'.format(count_pass))
            res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, padding='max_length',
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              truncation=True,
                                              return_tensors='pt',
                                              )

            self.input_ids, self.attention_mask, self.token_type_ids = res['input_ids'], res['attention_mask'], res[
                'token_type_ids']

            if has_label:
                self.labels = torch.tensor(labels).long()
                res['labels'] = self.labels

            print("Saving data to cached file %s", cached_file)
            torch.save(res, cached_file)
            print(self.input_ids[0].shape)
            print('max_len: {}'.format(self.max_len))
            print(tokenizer.decode(self.input_ids[0].numpy().tolist()))
            print(self.input_ids[0].numpy().tolist())
            print(self.attention_mask[0].numpy().tolist())
            print(self.token_type_ids[0].numpy().tolist())
            if has_label:
                print(self.labels[0].numpy().tolist())
                print(self.labels[0].shape)
                print(self.num_class)
            print('>>>>>>>>>>>>>Dataset size: {}'.format(self.__len__()))

    def __len__(self):

        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        if self.has_label:
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]
        else:
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx]


def process_mwp_V1(raw_mwp: dict, label2id_or_value: dict, max_len: int):
    sentence = raw_mwp['T_question']
    if len(sentence) > (max_len - 2):
        return None, None, None, None

    result_sens = []
    result_labels = []
    all_num_postions = []

    num_codes = raw_mwp['num_codes']

    for kk, vv in raw_mwp['T_number_map'].items():
        if kk not in num_codes:

            postions_in_q = []
            for cindex, value in enumerate(sentence):
                if kk == value:
                    postions_in_q.append(cindex)
            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()
            for position in postions_in_q:
                label2id = label2id_or_value["None"]
                label_vector = [0] * len(label2id_or_value)
                label_vector[label2id] = 1
                label_vector = [position] + label_vector

                result_sens.append(sentence)
                result_labels.append(label_vector)
            all_num_postions += postions_in_q

        else:

            postions_in_q = []
            for cindex, value in enumerate(sentence):
                if kk == value:
                    postions_in_q.append(cindex)

            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()







            elif len(postions_in_q) == 1:
                position = postions_in_q[0]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector

                result_sens.append(sentence)
                result_labels.append(label_vector)


            else:
                position = postions_in_q[-1]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector
                result_sens.append(sentence)
                result_labels.append(label_vector)

                for pp in postions_in_q[:-1]:
                    label2id = label2id_or_value["None"]
                    label_vector = [0] * len(label2id_or_value)
                    label_vector[label2id] = 1
                    label_vector = [pp] + label_vector
                    result_sens.append(sentence)
                    result_labels.append(label_vector)

            all_num_postions += postions_in_q

    total_all_num_postions = []
    sens_lens = []
    for i in range(len(result_sens)):
        total_all_num_postions.append(all_num_postions)
        sens_lens.append(len(sentence))

    assert len(result_sens) == len(result_labels) == len(total_all_num_postions)

    return result_sens, result_labels, total_all_num_postions, sens_lens


def get_num_class(nnn: str):
    if nnn == '1':
        return chr(945)
    elif nnn == '3.14':
        return chr(946)
    elif nnn.find('+') != -1 and nnn.find('/') != -1:
        return chr(947)
    elif nnn.find('/100') != -1:
        return chr(948)
    elif nnn.find('/') != -1:
        return chr(949)
    elif nnn.find('.') != -1:
        return chr(950)
    elif len(nnn) == 1:

        return chr(951)
    elif len(nnn) == 2:

        return chr(952)
    else:
        return chr(953)


def get_new_sentence(raw_mwp: dict):
    sentence = raw_mwp['T_question']
    new_sentence = list(sentence)
    for kk, vv in raw_mwp['T_number_map'].items():
        char_nnn = get_num_class(vv)
        postions_in_q = []
        for cindex, value in enumerate(sentence):
            if kk == value:
                postions_in_q.append(cindex)
        if len(postions_in_q) == 0:
            print('Can not find num !!! Wrong !!!!!!')
            sys.exit()
        for position in postions_in_q:
            if ord(new_sentence[position]) < 945 or ord(new_sentence[position]) > 965:
                print('wrong!!!! replace not a number!!!!')
                print('wrong!!!! replace not a number!!!!')
                sys.exit()
            new_sentence[position] = char_nnn

    return ''.join(new_sentence)


def process_mwp_V2(raw_mwp: dict, label2id_or_value: dict, max_len: int):
    sentence = raw_mwp['T_question']
    sen_lennn = len(sentence)
    if sen_lennn > (max_len - 2):
        return None, None, None, None

    result_sens = []
    result_labels = []
    all_num_postions = []
    num_codes = raw_mwp['num_codes']

    new_sentence = get_new_sentence(raw_mwp)

    for kk, vv in raw_mwp['T_number_map'].items():

        if kk not in num_codes:

            postions_in_q = []
            for cindex, value in enumerate(sentence):
                if kk == value:
                    postions_in_q.append(cindex)

            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()

            for position in postions_in_q:
                label2id = label2id_or_value["None"]
                label_vector = [0] * len(label2id_or_value)
                label_vector[label2id] = 1
                label_vector = [position] + label_vector

                result_sens.append(new_sentence)
                result_labels.append(label_vector)

            all_num_postions += postions_in_q

        else:

            postions_in_q = []
            for cindex, value in enumerate(sentence):
                if kk == value:
                    postions_in_q.append(cindex)

            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()



            elif len(postions_in_q) == 1:
                position = postions_in_q[0]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector

                result_sens.append(new_sentence)
                result_labels.append(label_vector)


            else:
                position = postions_in_q[-1]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector

                result_sens.append(new_sentence)
                result_labels.append(label_vector)

                for pp in postions_in_q[:-1]:
                    label2id = label2id_or_value["None"]
                    label_vector = [0] * len(label2id_or_value)
                    label_vector[label2id] = 1
                    label_vector = [pp] + label_vector

                    result_sens.append(new_sentence)
                    result_labels.append(label_vector)

            all_num_postions += postions_in_q

    total_all_num_postions = []
    sens_lens = []
    for i in range(len(result_sens)):
        total_all_num_postions.append(all_num_postions)
        sens_lens.append(sen_lennn)

    return result_sens, result_labels, total_all_num_postions, sens_lens


def get_vocab(data_path: list):
    word_set = set()

    for path in data_path:
        with open(path, 'r', encoding='utf-8')as ff:
            data = json.load(ff)
            print('data: {}'.format(len(data)))
            for mwp in data:
                ques_words = set(list(mwp['T_question']))

                word_set = word_set.union(ques_words)
    print('total words in data: {}'.format(len(word_set)))
    with open('./vocab_train_mawps.txt', 'w', encoding='utf-8')as fout:
        for ww in word_set:
            fout.write(ww + '\n')

    return word_set


def get_vocab_mawps(data_path: list):
    word_set = []

    for path in data_path:
        with open(path, 'r', encoding='utf-8')as ff:
            data = json.load(ff)
            print('data: {}'.format(len(data)))
            for mwp in data:
                ques_words = mwp['T_question_2'].lower().split(' ')
                word_set = word_set + ques_words
    word_set = set(word_set)
    print('total words in data: {}'.format(len(word_set)))
    with open('./vocab_train_mawps_02.txt', 'w', encoding='utf-8')as fout:
        for ww in word_set:
            fout.write(ww + '\n')

    return word_set
