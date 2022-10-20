import argparse
import json
from src.test_new import test_for_mwp_slover_rnn_data_new
from data_process.data_process import trans_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--multi_fc", default=True, type=bool)
parser.add_argument("--has_label", default=True, type=bool)

parser.add_argument("--test_dev_max_len", default=192, type=int)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--fc_size', default=2048, type=int)
parser.add_argument('--rnn_layer', default=4, type=int)
parser.add_argument('--dropout', default=0.1, type=int)
parser.add_argument('--rnn_cell', default='LSTM', type=str)

parser.add_argument("--test_data_path",
                    default='../dataset/Math23K/math23k_test.json',
                    type=str)
parser.add_argument("--fc_path",
                    default=None,
                    type=str)
parser.add_argument("--test_model_path",
                    default='./output/SUMC_RNN_final_model/latest_model/last-model.bin',
                    type=str)

parser.add_argument("--vocab_path", default="./vocab_train_data_new.txt", type=str)
parser.add_argument("--label2id_path", default='../dataset/Math23K/my_num_codes_all.json',
                    type=str)
parser.add_argument("--gpu_device", default="5", type=str)
parser.add_argument("--use_multi_gpu", default=False, type=bool)


if __name__ == "__main__":
    args = parser.parse_args()
    args.use_new_replace = False
    args.use_attention = False

    processed_dataset_test, _ = trans_dataset(args.test_data_path)
    with open(args.label2id_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    args.num_labels = len(label2id_list)

    args.test_data_path = processed_dataset_test
    test_for_mwp_slover_rnn_data_new(args)
