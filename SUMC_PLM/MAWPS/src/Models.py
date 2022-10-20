import logging
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel

from torch.nn.functional import mse_loss

logger = logging.getLogger(__name__)


class Batch_Net_large(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net_large, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.LeakyReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.LeakyReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MwpBertModel(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, train_loss='MSE', fc_path=None, multi_fc=False,
                 fc_hidden_size: int = 1024):
        super(MwpBertModel, self).__init__()
        self.num_labels = num_labels

        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)
        self.dropout = torch.nn.Dropout(0.1)

        if multi_fc:
            self.fc = Batch_Net_large(self.bert.config.hidden_size, fc_hidden_size, int(fc_hidden_size / 2), num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))

        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

        assert train_loss in ['MSE', 'L1', 'Huber']
        if train_loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif train_loss == 'L1':

            self.loss_func = torch.nn.L1Loss()
        else:
            self.loss_func = torch.nn.SmoothL1Loss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, getlogit: bool = False):
        token_embeddings, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)

        token_embeddings = token_embeddings[:, 1:-1, :]
        sen_vectors = []
        new_labels = labels[:, 1:].float()

        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
            sen_vectors.append(num_vector)

        sen_vectors = self.dropout(torch.cat(sen_vectors, 0))
        logits = self.fc(sen_vectors)

        if labels is not None:
            loss = self.loss_func(logits, new_labels)

            return logits, loss.mean()
        return logits, None

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))

    def get_sens_vec(self, sens: list):
        self.bert.eval()
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt", max_length=self.max_length)
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]

        logger.info("input ids shape: {},{}".format(input_ids.shape[0], input_ids.shape[1]))
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)

        all_sen_vec = []
        with torch.no_grad():
            for idx, batch_data in enumerate(data_loader):
                logger.info("get sentences vector: {}/{}".format(idx + 1, len(data_loader)))
                batch_data = [i.to(self.device) for i in batch_data]
                token_embeddings, pooler_output = self.bert(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                            token_type_ids=batch_data[2])
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":

                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)

                all_sen_vec.append(sen_vec.to("cpu").numpy())
        self.bert.train()
        return np.vstack(all_sen_vec)


class Batch_Net_CLS(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net_CLS, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.LeakyReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.LeakyReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MwpBertModel_CLS(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, train_loss='MSE', fc_path=None, multi_fc=False,
                 fc_hidden_size: int = 1024):
        super(MwpBertModel_CLS, self).__init__()
        self.num_labels = num_labels

        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)
        self.dropout = torch.nn.Dropout(0.1)

        if multi_fc:
            self.fc = Batch_Net_CLS(self.bert.config.hidden_size * 2, fc_hidden_size, int(fc_hidden_size / 2),
                                    num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size * 2, out_features=num_labels, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))

        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

        assert train_loss in ['MSE', 'L1', 'Huber']
        if train_loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif train_loss == 'L1':

            self.loss_func = torch.nn.L1Loss()
        else:
            self.loss_func = torch.nn.SmoothL1Loss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, getlogit: bool = False):
        token_embeddings, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)

        token_embeddings = token_embeddings[:, 1:-1, :]
        sen_vectors = []

        new_labels = labels[:, 1:].float()

        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
            sen_vectors.append(num_vector)
        sen_vectors = torch.cat(sen_vectors, 0)
        sen_vectors = torch.cat([sen_vectors, pooler_output], 1)
        sen_vectors = self.dropout(sen_vectors)

        logits = self.fc(sen_vectors)

        if labels is not None:
            loss = self.loss_func(logits, new_labels)
            return logits, loss.mean()
        return logits, None

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))

    def get_sens_vec(self, sens: list):
        self.bert.eval()
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt", max_length=self.max_length)
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]

        logger.info("input ids shape: {},{}".format(input_ids.shape[0], input_ids.shape[1]))
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)

        all_sen_vec = []
        with torch.no_grad():
            for idx, batch_data in enumerate(data_loader):
                logger.info("get sentences vector: {}/{}".format(idx + 1, len(data_loader)))
                batch_data = [i.to(self.device) for i in batch_data]
                token_embeddings, pooler_output = self.bert(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                            token_type_ids=batch_data[2])
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":

                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)

                all_sen_vec.append(sen_vec.to("cpu").numpy())
        self.bert.train()
        return np.vstack(all_sen_vec)
