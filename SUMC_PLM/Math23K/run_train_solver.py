import argparse
import json
import os
import torch
from src.train_solver import train_fast
from data_process.data_process import trans_dataset
parser = argparse.ArgumentParser()

parser.add_argument("--train_type", default='one-by-one-random', type=str,
                    help='one-by-one-in-same-batch or one-by-one-random or together')
parser.add_argument('--re_process_train_data', default=False, type=bool)
parser.add_argument("--deal_data_imbalance", default=0, type=int)
parser.add_argument("--multi_fc", default=True, type=bool)
parser.add_argument("--fc_hidden_size", default=2048, type=int)
parser.add_argument("--use_cls", default=True, type=bool)
parser.add_argument("--train_max_len", default=192, type=int)
parser.add_argument("--test_dev_max_len", default=192, type=int)
parser.add_argument("--use_new_token_type_id", default=True, type=bool)
parser.add_argument("--train_loss", default='MSE', type=str, help='MSE or L1 or Huber')

parser.add_argument("--use_multi_gpu", default=False, type=bool)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=150, type=int)

parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup", default=0.1, type=float)

parser.add_argument("--fc_path", default=None, type=str)
parser.add_argument("--pretrain_model_path",
                    default="path of the chinese_roberta_base",
                    type=str)

parser.add_argument("--train_data_path",
                    default="../../dataset/math23k_train.json",
                    type=str)

parser.add_argument("--dev_data_path",
                    default="../../dataset/math23k_test.json",
                    type=str)

parser.add_argument("--gpu_device", default="7", type=str)
parser.add_argument("--output_dir", default="./output/", type=str)
parser.add_argument("--model_name", default="test", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = args.__dict__
    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'setting_parameters.json'), 'w', encoding='utf-8')as f:
        f.write(json.dumps(args_dict, ensure_ascii=False, indent=2))

    processed_dataset_train, num_codes_list = trans_dataset(args.train_data_path)
    processed_dataset_test, _ = trans_dataset(args.dev_data_path)

    data_save = {}
    data_save['train'] = processed_dataset_train
    data_save['test'] = processed_dataset_test
    data_save['codes'] = num_codes_list
    torch.save(data_save, './save_data.bin')

    # data_save = torch.load('./save_data.bin')
    # processed_dataset_train, num_codes_list = data_save['train'], data_save['codes']
    # processed_dataset_test = data_save['test']

    args.num_labels = len(num_codes_list)
    print(args.num_labels)
    print(num_codes_list[:5])
    args.label2id_path = num_codes_list
    args.train_data_path = processed_dataset_train
    args.dev_data_path = processed_dataset_test

    train_fast(args)
