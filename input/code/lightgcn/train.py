import pandas as pd
import torch
from config import CFG, logging_conf, sweep_conf
from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, train
from lightgcn.utils import class2dict, get_logger, setSeeds

import wandb

# if CFG.user_wandb:
#     import wandb

#     wandb.init(**CFG.wandb_kwargs, config=class2dict(CFG))

setSeeds(CFG.seed)
logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


def main():
    logger.info("Task Started")

    logger.info("[1/1] Data Preparing - Start")
    # n_node = len(user+item)
    train_data, valid_data, test_data, n_node = prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/1] Data Preparing - Done")

    logger.info("[2/2] Model Building - Start")

    # if CFG.user_wandb:
    #     wandb.watch(model)

    logger.info("[3/3] Model Training - Start")

    if CFG.sweep:
        def runner():
            wandb.init(config=class2dict(CFG))
            w_config = wandb.config

            model = build(
                n_node,
                embedding_dim=w_config.hidden_dim,
                num_layers=w_config.n_layers,
                alpha=w_config.alpha,
                logger=logger.getChild("build"),
                **CFG.build_kwargs
            )
            model.to(device)

            wandb.watch(model)
            
            train(
                model,
                train_data,
                valid_data,
                n_epochs=w_config.n_epochs,
                learning_rate=w_config.learning_rate,
                use_wandb=w_config.user_wandb,
                weight=w_config.weight_basepath,
                logger=logger.getChild("train"),
            )

        sweep_id = wandb.sweep(sweep_conf, entity="egsbj", project="lightgcn")
        wandb.agent(sweep_id, runner, count=CFG.sweep_count)

    else:
        wandb.init(config=class2dict(CFG), entity="egsbj", project="lightgcn")

        model = build(
            n_node,
            embedding_dim=CFG.hidden_dim,
            num_layers=CFG.n_layers,
            alpha=CFG.alpha,
            logger=logger.getChild("build"),
            **CFG.build_kwargs
        )
        model.to(device)

        wandb.watch(model)
        
        train(
            model,
            train_data,
            valid_data,
            n_epochs=CFG.n_epochs,
            learning_rate=CFG.learning_rate,
            use_wandb=CFG.user_wandb,
            weight=CFG.weight_basepath,
            logger=logger.getChild("train"),
        )

    logger.info("[2/2] Model Building - Done")

    logger.info("[3/3] Model Training - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    main()
