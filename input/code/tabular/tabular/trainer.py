import math
import os

import numpy as np
import torch
import wandb
import lightgbm as lgb
from wandb.lightgbm import wandb_callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import joblib

# from .criterion import get_criterion
# from .dataloader import get_loaders
# from .metric import get_metric
# from .model import LSTM, LSTMATTN, Bert
# from .optimizer import get_optimizer
# from .scheduler import get_scheduler


def run(args, train_data, valid_data):
    # train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # # only when using warmup scheduler
    # args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
    #     args.n_epochs
    # )
    # args.warmup_steps = args.total_steps // 10

    wandb.init(project="DKT", entity="egsbj", config=vars(args), name=f"{args.model}")

    # model = get_model(args)
    if args.model == 'lightgbm':
        
        FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

        # x, y split
        print('x, y split')
        lgb_train, lgb_valid, answer = get_lgb_data(train_data, valid_data, FEATS)

        model = lgb.train(
        {'objective': args.objective}, 
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        verbose_eval=args.verbose_eval,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        callbacks=[wandb_callback()]
    )
        
    model_predict(model, valid_data, answer, FEATS)

    save_model(args, model)
    # model = get_model(args)
    # optimizer = get_optimizer(model, args)
    # scheduler = get_scheduler(optimizer, args)

    # best_auc = -1
    # early_stopping_counter = 0
    # for epoch in range(args.n_epochs):

    #     print(f"Start Training: Epoch {epoch + 1}")

    #     ### TRAIN
    #     train_auc, train_acc, train_loss = train(
    #         train_loader, model, optimizer, scheduler, args
    #     )

    #     ### VALID
    #     auc, acc = validate(valid_loader, model, args)

    #     ###  model save or early stopping
    #     wandb.log(
    #         {
    #             "epoch": epoch,
    #             "train_loss": train_loss,
    #             "train_auc": train_auc,
    #             "train_acc": train_acc,
    #             "valid_auc": auc,
    #             "valid_acc": acc,
    #         }
    #     )
    #     if auc > best_auc:
    #         best_auc = auc
    #         # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
    #         model_to_save = model.module if hasattr(model, "module") else model
    #         save_checkpoint(
    #             {
    #                 "epoch": epoch + 1,
    #                 "state_dict": model_to_save.state_dict(),
    #             },
    #             args.model_dir,
    #             "model.pt",
    #         )
    #         early_stopping_counter = 0
    #     else:
    #         early_stopping_counter += 1
    #         if early_stopping_counter >= args.patience:
    #             print(
    #                 f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
    #             )
    #             break

    #     # scheduler
    #     if args.scheduler == "plateau":
    #         scheduler.step(best_auc)

def get_lgb_data(train, test, FEATS):

    # X, y 값 분리
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)

    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_test = lgb.Dataset(test[FEATS], y_test)

    return lgb_train, lgb_test, y_test

def model_predict(model, valid_data, answer, FEATS):
    # 사용할 Feature 설정

    preds = model.predict(valid_data[FEATS])
    acc = accuracy_score(answer, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(answer, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')

# def train(train_loader, model, optimizer, scheduler, args):
#     model.train()

#     total_preds = []
#     total_targets = []
#     losses = []
#     for step, batch in enumerate(train_loader):
#         input = process_batch(batch, args)
#         preds = model(input)
#         targets = input[3]  # correct

#         loss = compute_loss(preds, targets)
#         update_params(loss, model, optimizer, scheduler, args)

#         if step % args.log_steps == 0:
#             print(f"Training steps: {step} Loss: {str(loss.item())}")

#         # predictions
#         preds = preds[:, -1]
#         targets = targets[:, -1]

#         if args.device == "cuda":
#             preds = preds.to("cpu").detach().numpy()
#             targets = targets.to("cpu").detach().numpy()
#         else:  # cpu
#             preds = preds.detach().numpy()
#             targets = targets.detach().numpy()

#         total_preds.append(preds)
#         total_targets.append(targets)
#         losses.append(loss)

#     total_preds = np.concatenate(total_preds)
#     total_targets = np.concatenate(total_targets)

#     # Train AUC / ACC
#     auc, acc = get_metric(total_targets, total_preds)
#     loss_avg = sum(losses) / len(losses)
#     print(f"TRAIN AUC : {auc} ACC : {acc}")
#     return auc, acc, loss_avg


# def validate(valid_loader, model, args):
#     model.eval()

#     total_preds = []
#     total_targets = []
#     for step, batch in enumerate(valid_loader):
#         input = process_batch(batch, args)

#         preds = model(input)
#         targets = input[3]  # correct

#         # predictions
#         preds = preds[:, -1]
#         targets = targets[:, -1]

#         if args.device == "cuda":
#             preds = preds.to("cpu").detach().numpy()
#             targets = targets.to("cpu").detach().numpy()
#         else:  # cpu
#             preds = preds.detach().numpy()
#             targets = targets.detach().numpy()

#         total_preds.append(preds)
#         total_targets.append(targets)

#     total_preds = np.concatenate(total_preds)
#     total_targets = np.concatenate(total_targets)

#     # Train AUC / ACC
#     auc, acc = get_metric(total_targets, total_preds)

#     print(f"VALID AUC : {auc} ACC : {acc}\n")

#     return auc, acc


def inference(args, test_data):

    # model = load_model(args)
    # model.eval()
    # _, test_loader = get_loaders(args, None, test_data)

    # total_preds = []

    # for step, batch in enumerate(test_loader):
    #     input = process_batch(batch, args)

    #     preds = model(input)

    #     # predictions
    #     preds = preds[:, -1]

    #     if args.device == "cuda":
    #         preds = preds.to("cpu").detach().numpy()
    #     else:  # cpu
    #         preds = preds.detach().numpy()

    #     total_preds += list(preds)

    
    model = load_model(args)

    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']
    # MAKE PREDICTION
    total_preds = model.predict(test_data[FEATS])

    write_path = os.path.join(args.output_dir, "submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


# def get_model(args):
#     """
#     Load model and move tensors to a given devices.
#     """
#     # if args.model == "lstm":
#     #     model = LSTM(args)
#     # if args.model == "lstmattn":
#     #     model = LSTMATTN(args)
#     # if args.model == "bert":
#     #     model = Bert(args)

#     # model.to(args.device)

#     # if args.model == 'lightgbm':
#     #     model = lgb.train(
#     #     {'objective': f'{args.objective}'}, 
#     #     train_data,
#     #     valid_sets=[train_data, valid_data],
#     #     verbose_eval=f'{args.verbose_eval}',
#     #     num_boost_round=f'{args.num_boost_round}',
#     #     early_stopping_rounds=f'{args.early_stopping_rounds}',
#     #     callbacks=[wandb_callback()]
#     # )

#     return model


# 배치 전처리
def process_batch(batch, args):

    test, question, tag, correct, mask = batch

    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)

    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)

    return (test, question, tag, correct, mask, interaction)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    # load_state = torch.load(model_path)
    # model = get_model(args)
    model = joblib.load(model_path)

    # # load model state
    # model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model

def save_model(args, model):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Saving Model from:", model_path)

    joblib.dump(model, model_path)