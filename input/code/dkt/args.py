import argparse
from xmlrpc.client import boolean

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )
    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 데이터 증강 (Data Augmentation)
    parser.add_argument(
        "--augmentation", action='store_true', help="data augmentation"
    )
    parser.add_argument(
        "--window", action='store_true', help="sliding window"
    )
    parser.add_argument(
        "--shuffle", action='store_true', help="sliding window shuffle"
    )
    parser.add_argument(
        "--shuffle_n", default=2, type=int, help="data augmentation count"
    )

    # config['window'] = False
    # config['stride'] = config['max_seq_len']
    # config['shuffle'] = False
    # config['shuffle_n'] = 2

    # 모델
    parser.add_argument(
        "--hidden_dim", default=1024, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=16, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # T Fixup
    parser.add_argument(
        "--Tfixup", action='store_true', help="Tfixup"
    )
    parser.add_argument("--layer_norm", action='store_false', help="layer norm")

#     config['Tfixup'] = False
#     config['layer_norm'] = True

    # 훈련
    parser.add_argument("--n_epochs", default=150, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=10, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstm", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )
    
    # Sweep 파라미터
    parser.add_argument("--sweep", action="store_true", help="sweep type")
    parser.add_argument("--sweep_count", default=10, type=int, help="sweep count")
    parser.add_argument("--sweep_name", type=str, default='DKT_LSTM', help='DKT_model-name' )
    parser.add_argument("--sweep_method", type=str, default='bayes', help='grid, random, bayes' )

    args = parser.parse_args()
    return args
