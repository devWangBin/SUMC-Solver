import argparse
import json
import os
import sys
from src.train_fast import train_fast
from data_process.data_process import trans_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=150, type=int)
parser.add_argument("--log_steps_per_epoch", default=10, type=int)
parser.add_argument("--lr", default=0.002, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--warmup", default=0.1, type=float)

parser.add_argument("--train_type", default='one-by-one-random', type=str,
                    help='one-by-one-in-same-batch or one-by-one-random or together')

parser.add_argument("--multi_fc", default=True, type=bool)
parser.add_argument("--train_max_len", default=192, type=int)
parser.add_argument("--test_dev_max_len", default=192, type=int)

parser.add_argument("--use_multi_gpu", default=False, type=bool)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--fc_size', default=2048, type=int)
parser.add_argument('--rnn_layer', default=4, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--rnn_cell', default='LSTM', type=str)

parser.add_argument("--fc_path", default=None, type=str)
parser.add_argument("--train_model_path",
                    default=None,
                    type=str)

parser.add_argument("--vocab_path", default="./vocab_train_data_new.txt", type=str)
parser.add_argument("--train_data_path",
                    default="../../dataset/math23k_train.json",
                    type=str)

parser.add_argument("--dev_data_path",
                    default="../../dataset/math23k_test.json",
                    type=str)

parser.add_argument("--gpu_device", default="2", type=str)
parser.add_argument("--output_dir", default="./output/", type=str)
parser.add_argument("--model_name", default="SUMC_RNN_final_model22", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    args.use_new_replace = False
    args.use_attention = False

    args_dict = args.__dict__
    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'setting_parameters.json'), 'w', encoding='utf-8')as f:
        f.write(json.dumps(args_dict, ensure_ascii=False, indent=2))

    processed_dataset_train, num_codes_list = trans_dataset(args.train_data_path)
    processed_dataset_test, _ = trans_dataset(args.dev_data_path)

    args.num_labels = len(num_codes_list)
    args.label2id_path = num_codes_list

    args.train_data_path = processed_dataset_train
    args.dev_data_path = processed_dataset_test

    train_fast(args)
