# ====================================================
# CFG
# ====================================================
class CFG:
    # Custom
    model = 'lightgcn' # 추후에 모델 추가시 모델별 메소드 생성해주세요!
    
    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="DKT", entity="egsbj", name = model)

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    hidden_dim = 64  # int
    n_layers = 1  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    seed = 42
    n_epoch = 20
    learning_rate = 0.001
    n_epochs = 20
    lr = 0.001
    weight_basepath = "./weight"



logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
