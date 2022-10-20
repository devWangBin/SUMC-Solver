import logging
import os

import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
from torch.nn import LayerNorm
import torch.nn as nn
import torch.nn.functional as F

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


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, fc_size, rnn_cell: str,
                 num_labels, fc_path=None, multi_fc=True, drop_p=0.1):
        super(RNNModel, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        print(type(drop_p))
        print('drop_p: {}'.format(drop_p))

        if rnn_cell == 'LSTM':
            self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                  batch_first=True, num_layers=num_layer, dropout=drop_p,
                                  bidirectional=True)
        elif rnn_cell == 'GRU':
            self.bilstm = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layer,
                                 dropout=drop_p, bidirectional=True,
                                 batch_first=True)

        self.dropout = SpatialDropout(drop_p)
        self.dropout_fc = torch.nn.Dropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)

        if multi_fc:

            self.fc = Batch_Net_large(hidden_size * 2, fc_size, int(fc_size / 2), num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=hidden_size * 2, out_features=num_labels, bias=True)

        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))

    def forward(self, inputs_ids, input_mask, labels):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)

        token_embeddings = seqence_output[:, 1:-1, :]
        sen_vectors = []
        new_labels = labels[:, 1:].float()

        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
            sen_vectors.append(num_vector)

        sen_vectors = self.dropout_fc(torch.cat(sen_vectors, 0))
        logits = self.fc(sen_vectors)
        loss = torch.nn.functional.mse_loss(logits, new_labels)

        return logits, loss.mean()

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()  # 1 x B x H*2
        repeat_dims[0] = max_len  # seq_len, 1, 1
        hidden = hidden.repeat(*repeat_dims)  # S x B x H*2
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class RNNModel_ATT(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, fc_size, rnn_cell: str,
                 num_labels, fc_path=None, multi_fc=True, drop_p=0.1):
        super(RNNModel_ATT, self).__init__()

        self.hidden_size = hidden_size
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        if rnn_cell == 'LSTM':
            self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                  batch_first=False, num_layers=num_layer, dropout=drop_p,
                                  bidirectional=True)
        elif rnn_cell == 'GRU':
            self.bilstm = nn.GRU(embedding_size, hidden_size, num_layer, dropout=drop_p, bidirectional=True,
                                 batch_first=False)

        # dropout setting
        self.em_dropout = SpatialDropout(drop_p)
        self.dropout_fc = torch.nn.Dropout(0.1)

        self.attn = Attn(hidden_size * 2)

        self.layer_norm = LayerNorm(hidden_size * 2)
        if multi_fc:
            self.fc = Batch_Net_large(hidden_size * 4, fc_size, int(fc_size / 2), num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=hidden_size * 4, out_features=num_labels, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))

    def forward(self, inputs_ids, input_mask, labels):
        embs = self.embedding(inputs_ids)
        embs = self.em_dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs.transpose(0, 1))
        seqence_output = self.layer_norm(seqence_output)
        seqence_output = seqence_output[1:-1, :, :]
        input_mask = input_mask[:, 1:-1]
        sen_vectors = []
        new_labels = labels[:, 1:].float()

        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(seqence_output[lll[0], idd, :], 0)
            sen_vectors.append(num_vector)
        sen_vectors = torch.cat(sen_vectors, 0)

        attn_weights = self.attn(sen_vectors.unsqueeze(0), seqence_output, input_mask)
        context = attn_weights.bmm(seqence_output.transpose(0, 1))  # B x S=1 x N
        context = context.squeeze(1)  # B x H*2
        sen_vectors = torch.cat((sen_vectors, context), 1)

        logits = self.fc(sen_vectors)
        loss = torch.nn.functional.mse_loss(logits, new_labels)
        return logits, loss.mean()

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))
