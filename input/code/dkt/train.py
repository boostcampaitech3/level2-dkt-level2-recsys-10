import os

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds

import config

def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(test_file_name= args.test_file_name, train_file_name= args.file_name)
    train_data = preprocess.get_train_data()
    valid_data = preprocess.get_valid_data()
    
    # run the sweep
    if args.sweep:
        def runner():
            wandb.init(config=vars(args))
            w_config = wandb.config
            trainer.run(w_config, train_data, valid_data)

        sweep_id = wandb.sweep(config.sweep_config, entity="egsbj", project="DKT")
        wandb.agent(sweep_id, runner, count=args.sweep_count)
    
    else:
        wandb.init(config=vars(args), name=args.model, entity="egsbj", project="DKT")
        trainer.run(args, train_data, valid_data)

if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
