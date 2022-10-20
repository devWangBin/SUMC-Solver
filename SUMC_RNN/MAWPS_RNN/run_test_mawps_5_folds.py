import argparse
import json
from src_mawps.test_new import test_for_mwp_slover_rnn_data_new

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--multi_fc", default=True, type=bool)
parser.add_argument("--has_label", default=True, type=bool)

parser.add_argument("--num_labels", default=28, type=int)
parser.add_argument("--test_dev_max_len", default=100, type=int)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--fc_size', default=2048, type=int)
parser.add_argument('--rnn_layer', default=4, type=int)
parser.add_argument('--dropout', default=0.1, type=int)
parser.add_argument('--rnn_cell', default='LSTM', type=str)

parser.add_argument("--fc_path",
                    default=None,
                    type=str)

parser.add_argument("--test_model_path",
                    default='./output/madel_save_name',
                    type=str)

parser.add_argument("--vocab_path", default="./vocab_train_mawps.txt", type=str)

parser.add_argument("--label2id_path", default='../../dataset/codes_mawps.json',
                    type=str)

parser.add_argument("--gpu_device", default="0", type=str)
parser.add_argument("--use_multi_gpu", default=False, type=bool)

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

        args = parser.parse_args()
        args.use_new_replace = False
        args.use_attention = False

        args.test_data_path = pairs_tested
        model_path = '/latest_model/last-model.bin'

        print(args.test_model_path)
        args.test_model_path = str(args.test_model_path + '_' + str(fold) + model_path)
        print(args.test_model_path)
        print('fold: {}'.format(fold))
        test_for_mwp_slover_rnn_data_new(args)
