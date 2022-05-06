import os

import torch
import wandb
from args import parse_args
from tabular import trainer
from tabular.dataloader import Preprocess
from tabular.utils import setSeeds

import config

def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)

    print('[STEP 1] Load the data') # train, test data 전부 로드
    preprocess.load_train_data(test_file_name= args.test_file_name, train_file_name= args.file_name)

    print('[STEP 2] Preprocess data and Split data into Train and Valid') # FE 포함하여 데이터 전처리 후 train valid split
    train_data = preprocess.get_train_data()
    valid_data = preprocess.get_valid_data()

    print("[STEP 3] Convert to data satisfying choosed model's style") # model에 맞는 데이터 스타일로 변경
    train_data, valid_data, X_valid, y_valid = preprocess.convert_dataset(train_data, valid_data)
    args.FEATS = preprocess.FEATS

    print("[STEP 4] Train the Model")
    if args.sweep:
        def runner():
            wandb.init(config=vars(args))
            trainer.run(wandb.config, train_data, valid_data, X_valid, y_valid)

        sweep_id = wandb.sweep(config.sweep_config, entity="egsbj", project="tabular")
        wandb.agent(sweep_id, runner, count=args.sweep_count)

    else:
        wandb.init(config=vars(args), name=args.model, entity="egsbj", project="tabular")
        trainer.run(args, train_data, valid_data, X_valid, y_valid)

if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
