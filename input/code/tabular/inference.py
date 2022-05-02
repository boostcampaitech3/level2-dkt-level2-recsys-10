import os

import torch
from args import parse_args
from tabular import trainer
from tabular.dataloader import Preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    print('load_test_data')
    preprocess.load_test_data(args.test_file_name)
    print('get test data from preprocess class')
    test_data = preprocess.get_test_data()

    print('inference')
    trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
