import os

import torch
from args import parse_args
from tabular import trainer
from tabular.dataloader import Preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    print('[STEP 1] Load the data') # train, test data 전부 로드
    preprocess = Preprocess(args)
    preprocess.load_test_data(test_file_name= args.test_file_name, train_file_name= args.file_name)

    print('[STEP 2] Preprocess test data satisfying')
    test_data = preprocess.get_test_data()
    
    print('[STEP 3] Inference')
    trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
