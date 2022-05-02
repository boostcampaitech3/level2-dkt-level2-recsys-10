import os

import torch
import wandb
from args import parse_args
from tabular import trainer
from tabular.dataloader import Preprocess
from tabular.utils import setSeeds


def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)

    print('load_train_data')
    preprocess.load_train_data(args.file_name)
    print('get train data from preprocess class')
    train_data = preprocess.get_train_data()

    # train, valid split
    print('train, valid split')
    train_data, valid_data = preprocess.split_data(train_data)

    # wandb.init(project="tabular", config=vars(args))
    print('train')
    trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
