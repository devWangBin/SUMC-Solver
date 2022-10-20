import os
import sys
import json
import torch
from transformers import BertTokenizer
import random
from tqdm import tqdm


class MWPDatasetLoader(object):
    def __init__(self, data, batch_size, shuffle, tokenizer: BertTokenizer, seed, sort=False):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.sort = sort
        self.tokenizer = tokenizer
        # self.label2id = label2id
        self.reset()

    def reset(self, doshuffle: bool = False):
        self.examples = self.preprocess(self.data)
        print('total data len: {}'.format(len(self.examples)))
        if self.sort:
            self.examples = sorted(self.examples, key=lambda x: x[2], reverse=True)
        if self.shuffle or doshuffle:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        self.features = [self.examples[i:i + self.batch_size] for i in range(0, len(self.examples), self.batch_size)]
        print(f"{len(self.features)} batches created")

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        #  (sentence_list, num_codes_labels, all_num_postions) = d
        for d in data:
            sen_tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + d[0] + ['[SEP]'])
            x_len = len(sen_tokens)
            # 准备token_type_id
            token_type_id = [0] * x_len
            for pp in d[2]:
                token_type_id[pp + 1] = 1
            for label in d[1]:
                processed.append((sen_tokens, label, x_len, token_type_id))
        return processed

    def get_long_tensor(self, tokens_list, batch_size, mask=None):
        """ Convert list of list of tokens to a padded LongTensor. """
        token_len = max(len(x) for x in tokens_list)
        tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_ = torch.LongTensor(batch_size, token_len).fill_(0)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
            if mask:
                mask_[i, :len(s)] = torch.tensor([1] * len(s), dtype=torch.long)
        if mask:
            return tokens, mask_
        return tokens

    def sort_all(self, batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """
        # [[1, 2, 3], [4, 5, 6]]
        # [(1, 4), (2, 5), (3, 6)]
        # [(1, 2, 3), (4, 5, 6)]
        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    def __len__(self):
        # return 50
        return len(self.features)

    def __getitem__(self, index):
        """ Get a batch with index. """
        if not isinstance(index, int):
            raise TypeError
        if index < 0 or index >= len(self.features):
            raise IndexError
        # batch list...
        batch = self.features[index]
        batch_size = len(batch)
        # zip(*) 前面的list不会加一层list
        batch = list(zip(*batch))
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = self.sort_all(batch, lens)
        chars = batch[0]
        input_ids, input_mask = self.get_long_tensor(chars, batch_size, mask=True)
        label_ids = self.get_long_tensor(batch[1], batch_size)
        token_type_ids = self.get_long_tensor(batch[3], batch_size)
        # input_lens = [len(x) for x in batch[0]]
        return (input_ids, input_mask, token_type_ids, label_ids)


def process_dataset(file, label2id_data_path: str, max_len: int, lower: bool):
    # 加载类别标签，形成label2id的字典
    label2id_or_value = {}
    with open(label2id_data_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde

    # 加载输入数据
    processed_dataset = []
    if isinstance(file, list):
        dataset = file
    elif isinstance(file, str):
        with open(file, 'r', encoding='utf-8') as fr:
            dataset = json.load(fr)
    else:
        print('file parameter wrong !!! \n exit !!!')
        sys.exit()

    print('total input dataset len: {}'.format(len(dataset)))
    # 开始处理数据
    passed_count = 0
    pbar = tqdm(dataset)
    for idd, raw_mwp in enumerate(pbar):
        # max_len 最大模型输入长度，不是batch的输入长度
        pppp_dadd = process_one_mawps(raw_mwp, label2id_or_value, max_len, lower=lower)
        if pppp_dadd is not None:
            processed_dataset.append(pppp_dadd)
        else:
            passed_count += 1
    print('after process dataset len: {}'.format(len(processed_dataset)))
    print('total passed: {}'.format(passed_count))
    print(processed_dataset[0])

    return processed_dataset


def process_one_mawps(raw_mwp: dict, label2id_or_value: dict, max_len: int, lower: bool):
    # if lower:
    #     # sentence_list = raw_mwp['T_question_2'].lower().split(' ')
    #     sentence_list = list(raw_mwp['T_question'].lower())
    # else:
    #     # sentence_list = raw_mwp['T_question_2'].split(' ')
    #     sentence_list = list(raw_mwp['T_question'])

    if lower:
        sentence_list = raw_mwp['T_question_2'].lower().split(' ')
    else:
        sentence_list = raw_mwp['T_question_2'].split(' ')
        sys.exit()

    if len(sentence_list) > (max_len - 2):
        return None
    # (sen_list, label_list, input_len)
    # datas_for_one_mwp = []
    num_codes_labels = []
    all_num_postions = []

    num_codes = raw_mwp['num_codes']
    for kk, vv in raw_mwp['T_number_map'].items():
        if kk not in num_codes:
            # 该数字为无关数字 每一个位置上的数字都将形成一条训练数据
            postions_in_q = []
            for cindex, value in enumerate(sentence_list):
                if kk == value:
                    postions_in_q.append(cindex)
            # 返回的时字符串，而不是index
            # postion_in_q = re.findall(kk, sentence)
            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()
            for position in postions_in_q:
                label2id = label2id_or_value["None"]
                label_vector = [0] * len(label2id_or_value)
                label_vector[label2id] = 1
                label_vector = [position] + label_vector
                # label_vector 的第一位为操作数在问题文本中的 position
                num_codes_labels.append(label_vector)
            all_num_postions += postions_in_q
        else:
            # 数字在表达式中，为相关数字
            postions_in_q = []
            for cindex, value in enumerate(sentence_list):
                if kk == value:
                    postions_in_q.append(cindex)

            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()

            # 若在q中有多个数字，多余的数字一般都是没什么意义的数字如301班，6月1号等
            # 策略，仅取最后出现的那个作为操作数，其他的认为时无关数据
            elif len(postions_in_q) == 1:
                position = postions_in_q[0]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector
                num_codes_labels.append(label_vector)
            # 题目中有多个该数字，只取最后出现的那个作为真实操作数 其它位置的看作是无关的
            else:
                position = postions_in_q[-1]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector
                # result_data.append((sentence, label_vector))
                num_codes_labels.append(label_vector)
                # 对于其它位置的
                for pp in postions_in_q[:-1]:
                    label2id = label2id_or_value["None"]
                    label_vector = [0] * len(label2id_or_value)
                    label_vector[label2id] = 1
                    label_vector = [pp] + label_vector
                    num_codes_labels.append(label_vector)

            all_num_postions += postions_in_q

    assert len(num_codes_labels) == len(all_num_postions)
    return (sentence_list, num_codes_labels, all_num_postions)


def process_one_mawps_for_test(raw_mwp: dict, label2id_or_value: dict, max_len: int, lower: bool,
                               tokenizer: BertTokenizer):
    data_mwp = process_one_mawps(raw_mwp, label2id_or_value, max_len, lower)
    # (sentence_list, num_codes_labels, all_num_postions) = data_mwp
    if data_mwp is None:
        return None

    processed = []
    sen_tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + data_mwp[0] + ['[SEP]'])

    x_len = len(sen_tokens)
    pad_len = 0
    if x_len < 100 and True:
        pad_len = (100 - x_len)
        sen_tokens = sen_tokens + [0] * pad_len
    x_len_new = len(sen_tokens)
    # 准备token_type_id
    token_type_id = [0] * x_len_new
    attention_mask = [1] * x_len + [0] * pad_len
    for pp in data_mwp[2]:
        token_type_id[pp + 1] = 1
    for label in data_mwp[1]:
        processed.append((sen_tokens, attention_mask, token_type_id, label))
    return processed
