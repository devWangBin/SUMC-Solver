import argparse

from src.Test import test_for_mwp_slover2
from src.Test_new import test_for_mwp_BERT_slover
from data_process.data_process import trans_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--multi_fc", default=True, type=bool)

parser.add_argument("--test_dev_max_len", default=192, type=int)
parser.add_argument("--use_cls", default=True, type=bool)

parser.add_argument("--test_data_path",
                    default='../../dataset/math23k_test.json',
                    type=str)
parser.add_argument("--train_data_path",
                    default="../../dataset/math23k_train.json",
                    type=str)

parser.add_argument("--pretrain_model_path_for_test",
                    default="./output/model_save/latest_model/",
                    type=str)

parser.add_argument("--gpu_device", default="7", type=str)

parser.add_argument("--use_new_token_type_id", default=True, type=bool)
parser.add_argument("--use_multi_gpu", default=False, type=bool)
parser.add_argument("--fc_hidden_size", default=2048, type=int)
parser.add_argument("--train_loss", default='MSE', type=str, help='MSE or L1 or Huber')

if __name__ == "__main__":

    args = parser.parse_args()

    processed_dataset_train, num_codes_list = trans_dataset(args.train_data_path)
    processed_dataset_test, _ = trans_dataset(args.test_data_path)

    args.num_labels = len(num_codes_list)
    args.label2id_path = num_codes_list
    # args.train_data_path = processed_dataset_train
    args.test_data_path = processed_dataset_test

    test_for_mwp_BERT_slover(args)
