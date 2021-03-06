import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data    

    def get_test_data(self):
        return self.test_data


    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = self.args.CAT_COLUMN

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

                df[col] = np.where(df[col].isin(le.classes_), df[col], "unknown")

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # def convert_time(s):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        # df.sort_values(by=['userID','Timestamp'], inplace=True)
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        
        #################################### 0.03
        # FE. 1 : 문제 푸는 시간 
        diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        df['elapsed'] = diff
        ####################################

        #################################### 0.003
        # FE. 2 : 시험지 대분류 (A000 쪼개서) 별 정답률 구하기 
        df['main_ca'] = df['testId'].str[:4]

        df['main_ca_correct_answer'] = df.groupby('main_ca')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['main_ca_total_answer'] = df.groupby('main_ca')['answerCode'].cumcount()
        df['main_ca_acc'] = df['main_ca_correct_answer']/df['main_ca_total_answer']
        ####################################

        ####################################
        # FE. 3 : 유저별 하나의 시험지를 다 푸는데 걸리는 시간
        # ⇒ 이상치 처리 후(마지막문제 0초, 오랜만에 푸는 문제 수만초 등) 유저별,시험지별 시간의 평균 
        normal_elapsed=df[(df['elapsed']!=0) & (df['elapsed']<660)]['elapsed']

        def normalize_outlier(x):
            if x>660:
                return np.random.choice(normal_elapsed)
            else:
                return x

        df['outliar_elapsed'] = df['elapsed'].apply(normalize_outlier)
        elapsed_test = df.groupby(['userID','testId']).mean()['outliar_elapsed']
        elapsed_test.name='elapsed_test'
        df = pd.merge(df,elapsed_test, on=['userID','testId'], how="left")
        ####################################

        ####################################
        ## FE. 4 : 과거에 푼 문제에 초점을 맞추자 => 과거 해당 문제 평균 정답률 => 약간의 성능 하락
        df['past_content_count'] = df.groupby(['userID', 'assessmentItemID']).cumcount()
        df['shift'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
        df['past_content_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()
        df.drop(['shift'], axis=1, inplace=True)
        # 과거 해당 문제 평균 정답률
        df['average_content_correct'] = (df['past_content_correct'] / df['past_content_count']).fillna(0)
        ####################################

        ####################################
        ## FE. 5 : 문항별(accessmentItemID) 정답률 => 성능 매우 높아짐
        correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
        correct_a.columns = ["assessment_mean", 'assessment_sum']

        df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
        ####################################

        ####################################
        ## FE. 6 : 시험지 많이 풀수록 맞출 확률이 높아진다? => 0.0007정도 성능 하락
        df['testCnt'] = df.groupby(['userID', 'testId']).cumcount()

        ####################################

        ####################################
        ## FE. 7 : 문제 풀이 시간대(hour) , 시간대별 정답률(correct_per_hour) 
        df['hour'] = df['Timestamp'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)
        hour_dict = df.groupby(['hour'])['answerCode'].mean().to_dict()
        df['correct_per_hour'] = df['hour'].map(hour_dict)
        ####################################

        ####################################
        ## FE. 8 : 태그 + 시험지 정답률
        correct_ka = df.groupby(['KnowledgeTag', 'testId'])['answerCode'].agg(['mean', 'sum'])
        correct_ka.columns = ["ka_accessment_mean", 'ka_accessment_sum']

        df = pd.merge(df, correct_ka, on=['testId', 'KnowledgeTag'], how="left")
        ####################################

        ####################################
        # base FE
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
        ####################################

        # TODO
        self.args.USERID_COLUMN = ['userID']
        self.args.CAT_COLUMN = ["assessmentItemID"] #, "testId", "KnowledgeTag"]
        self.args.CON_COLUMN = ['assessment_mean', 'assessment_sum', 'outliar_elapsed','past_content_count'] #['user_correct_answer', 'user_acc', "accessment_mean", 'accessment_sum']
        self.args.ANSWER_COLUMN = ['answerCode']
        
        return df

    def df_group_value_apply(self, r):
        return tuple([r[x].values for x in self.args.CAT_COLUMN] + [r[x].values for x in self.args.CON_COLUMN] + [r[x].values for x in self.args.ANSWER_COLUMN])

    def load_data_from_file(self, test_file_name, train_file_name=None, is_train=True):
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
            }        
        test_csv_file_path = os.path.join(self.args.data_dir, test_file_name)
        test_df = pd.read_csv(test_csv_file_path,dtype=dtype, parse_dates=['Timestamp'])
        
        if is_train:
            test_df = test_df.query("answerCode != -1")
            train_csv_file_path = os.path.join(self.args.data_dir, train_file_name)
            train_df = pd.read_csv(train_csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
            self.train_userID = train_df['userID'].unique().tolist()
            self.valid_userID = test_df['userID'].unique().tolist()
            df = pd.concat([train_df, test_df], axis=0)
        else :
            df = test_df

        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        # 인코딩된 mapper 정보 저장
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_embedding_layers = []       # 나중에 사용할 떄 embedding key들을 저장
        for val in self.args.CAT_COLUMN:
            self.args.n_embedding_layers.append(len(np.load(os.path.join(self.args.asset_dir, val+'_classes.npy'))))
            
        # self.args.n_questions = len(
        #     np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        # )
        # self.args.n_test = len(
        #     np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        # )
        # self.args.n_tag = len(
        #     np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        # )
        # 유저별 시간을 기준으로 sort
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        # columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        columns = self.args.USERID_COLUMN+self.args.CAT_COLUMN+self.args.CON_COLUMN+self.args.ANSWER_COLUMN
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                self.df_group_value_apply
            )
        )

        return group.values

    def load_train_data(self, test_file_name, train_file_name):
        data = self.load_data_from_file(test_file_name, train_file_name)
        self.valid_data = data[self.valid_userID]        
        self.train_data = data[self.train_userID]

    def load_test_data(self, test_file_name):
        self.test_data = self.load_data_from_file(test_file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # test, question, tag, correct = row[0], row[1], row[2], row[3]

        # cate_cols = [test, question, tag, correct]
        cate_cols = [val for val in row]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            # 모든 seq가 의미가 있으므로 모두 1로
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0]) # batch 길이
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    # seq 자리에 0(padding)을 채워주는 단계 여기서는 pre padding을 사용한다.
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate, # batch_size 만큼 뽑아주기위해서 padding을 해준다
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader

def slidding_window(data, args):
    window_size = args.max_seq_len
    args.stride = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def data_augmentation(data, args):
    if args.augmentation:
        data = slidding_window(data, args)

    return data