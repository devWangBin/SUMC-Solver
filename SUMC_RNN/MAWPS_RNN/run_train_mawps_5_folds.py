import argparse
import json
import os
import sys
from src_mawps.train_fast import train_fast

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--log_steps_per_epoch", default=10, type=int)
parser.add_argument("--lr", default=0.0008, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--warmup", default=0.1, type=float)

parser.add_argument("--train_type", default='one-by-one-random', type=str,
                    help='one-by-one-in-same-batch or one-by-one-random or together')

parser.add_argument("--num_labels", default=28, type=int)
parser.add_argument("--multi_fc", default=True, type=bool)
parser.add_argument("--train_max_len", default=100, type=int)
parser.add_argument("--test_dev_max_len", default=100, type=int)

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
parser.add_argument("--vocab_path", default="./vocab_train_mawps.txt", type=str)

parser.add_argument("--label2id_path", default='../../dataset/codes_mawps.json',
                    type=str)

parser.add_argument("--gpu_device", default="3", type=str)
parser.add_argument("--output_dir", default="./output/", type=str)
parser.add_argument("--model_name", default="model_save_name", type=str)

if __name__ == "__main__":
    with open('../../dataset/mawps_processed.json', 'r', encoding='utf-8')as fin:
        data = json.load(fin)
    data_train = data
    fold_size = int(len(data) * 0.2 + 1)

    fold_pairs = []
    for split_fold in range(4):
        fold_start = fold_size * split_fold
        fold_end = fold_size * (split_fold + 1)
        fold_pairs.append(data[fold_start:fold_end])
    fold_pairs.append(data[(fold_size * 4):])

    for fold in range(5):

        pairs_tested = []
        pairs_trained = []
        for fold_t in range(5):
            if fold_t == fold:
                pairs_tested += fold_pairs[fold_t]
            else:
                pairs_trained += fold_pairs[fold_t]
        args = None
        args = parser.parse_args()

        args.use_new_replace = False
        args.use_attention = False
        print(args.dropout)

        args.model_name = args.model_name + '_' + str(fold)
        args.train_data_path = pairs_trained

        args_dict = args.__dict__
        out_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'setting_parameters.json'), 'w', encoding='utf-8')as f:
            f.write(json.dumps(args_dict, ensure_ascii=False, indent=2))
        args.dev_data_path = pairs_tested
        train_fast(args)
