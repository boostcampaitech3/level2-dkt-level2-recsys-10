import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, df, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """

        users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
        
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(users)

        max_train_data_len = ratio*len(df)
        sum_of_train_data = 0
        user_ids = []

        for user_id, count in users:
            sum_of_train_data += count
            if max_train_data_len < sum_of_train_data:
                break
            user_ids.append(user_id)


        train = df[df['userID'].isin(user_ids)]
        test = df[df['userID'].isin(user_ids) == False]

        #test데이터셋은 각 유저의 마지막 interaction만 추출
        test = test[test['userID'] != test['userID'].shift(-1)]
        return train, test

        # size = int(len(data) * ratio)
        # data_1 = data[:size]
        # data_2 = data[size:]

        # return data_1, data_2

    # def get_lgb_data(self, train, test):
    #     # 사용할 Feature 설정
    #     FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
    #             'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

    #     # X, y 값 분리
    #     y_train = train['answerCode']
    #     train = train.drop(['answerCode'], axis=1)

    #     y_test = test['answerCode']
    #     test = test.drop(['answerCode'], axis=1)

    #     lgb_train = lgb.Dataset(train[FEATS], y_train)
    #     lgb_test = lgb.Dataset(test[FEATS], y_test)

    #     return lgb_train, lgb_test, y_test

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        # test에서만 적용되는 전처리
        if not is_train:
            # LEAVE LAST INTERACTION ONLY
            df = df[df['userID'] != df['userID'].shift(-1)]

            # DROP ANSWERCODE
            df = df.drop(['answerCode'], axis=1)

        return df

    def __feature_engineering(self, df):
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
        #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        

        # # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        # self.args.n_questions = len(
        #     np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        # )
        # self.args.n_test = len(
        #     np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        # )
        # self.args.n_tag = len(
        #     np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        # )

        # df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        # columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        # group = (
        #     df[columns]
        #     .groupby("userID")
        #     .apply(
        #         lambda r: (
        #             r["testId"].values,
        #             r["assessmentItemID"].values,
        #             r["KnowledgeTag"].values,
        #             r["answerCode"].values,
        #         )
        #     )
        # )

        # return group.values

        return df

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


# class DKTDataset(torch.utils.data.Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.args = args

#     def __getitem__(self, index):
#         row = self.data[index]

#         # 각 data의 sequence length
#         seq_len = len(row[0])

#         test, question, tag, correct = row[0], row[1], row[2], row[3]

#         cate_cols = [test, question, tag, correct]

#         # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
#         if seq_len > self.args.max_seq_len:
#             for i, col in enumerate(cate_cols):
#                 cate_cols[i] = col[-self.args.max_seq_len :]
#             mask = np.ones(self.args.max_seq_len, dtype=np.int16)
#         else:
#             mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
#             mask[-seq_len:] = 1

#         # mask도 columns 목록에 포함시킴
#         cate_cols.append(mask)

#         # np.array -> torch.tensor 형변환
#         for i, col in enumerate(cate_cols):
#             cate_cols[i] = torch.tensor(col)

#         return cate_cols

#     def __len__(self):
#         return len(self.data)


# from torch.nn.utils.rnn import pad_sequence


# def collate(batch):
#     col_n = len(batch[0])
#     col_list = [[] for _ in range(col_n)]
#     max_seq_len = len(batch[0][-1])

#     # batch의 값들을 각 column끼리 그룹화
#     for row in batch:
#         for i, col in enumerate(row):
#             pre_padded = torch.zeros(max_seq_len)
#             pre_padded[-len(col) :] = col
#             col_list[i].append(pre_padded)

#     for i, _ in enumerate(col_list):
#         col_list[i] = torch.stack(col_list[i])

#     return tuple(col_list)


# def get_loaders(args, train, valid):

#     pin_memory = False
#     train_loader, valid_loader = None, None

#     if train is not None:
#         trainset = DKTDataset(train, args)
#         train_loader = torch.utils.data.DataLoader(
#             trainset,
#             num_workers=args.num_workers,
#             shuffle=True,
#             batch_size=args.batch_size,
#             pin_memory=pin_memory,
#             collate_fn=collate,
#         )
#     if valid is not None:
#         valset = DKTDataset(valid, args)
#         valid_loader = torch.utils.data.DataLoader(
#             valset,
#             num_workers=args.num_workers,
#             shuffle=False,
#             batch_size=args.batch_size,
#             pin_memory=pin_memory,
#             collate_fn=collate,
#         )

#     return train_loader, valid_loader
