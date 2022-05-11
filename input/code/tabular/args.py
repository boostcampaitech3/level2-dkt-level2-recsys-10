import argparse


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
        "--model_name", default="model.pkl", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    # 모델 파라미터
    parser.add_argument("--model", default="lightgbm", type=str, help="model type")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="model learning_rate")

    parser.add_argument("--objective", default='binary', type=str)
    parser.add_argument("--verbose_eval", default=100, type=int)
    parser.add_argument("--num_boost_round", default=500, type=int)
    parser.add_argument("--early_stopping_rounds", default=200, type=int)
    parser.add_argument("--max_depth", default=10, type=int)

    # Sweep 파라미터
    parser.add_argument("--sweep", action="store_true", help="sweep type")
    parser.add_argument("--sweep_count", default=10, type=int, help="sweep count")
    parser.add_argument("--sweep_method", type=str, default='bayes', help='grid, random, bayes' )
    parser.add_argument("--sweep_name", type=str, default='test', help='grid, random, bayes' )
    parser.add_argument("--sweep_feats", action="store_true", help="sweep feats tune type")

    args = parser.parse_args()

    return args
