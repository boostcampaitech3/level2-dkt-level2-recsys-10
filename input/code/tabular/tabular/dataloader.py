import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import lightgbm as lgb
from catboost import Pool
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

                df[col] = np.where(df[col].isin(le.classes_), df[col], "unknown")

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

        # (2) FEATS는 FE가 직접적으로 작동이 되는 부분에서 언급되는것이 좋을것 같다.
        self.FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

        # TODO catboost는 Categorical columns name을 지정해줘야한다.
        #self.CATS = ['KnowledgeTag']
        self.CATS = []
        return df

    def load_data_from_file(self, test_file_name, train_file_name=None, is_train = True):
        test_csv_file_path = os.path.join(self.args.data_dir, test_file_name)
        test_df = pd.read_csv(test_csv_file_path)
        
        train_csv_file_path = os.path.join(self.args.data_dir, train_file_name)
        train_df = pd.read_csv(train_csv_file_path)
        self.train_userID = train_df['userID'].unique().tolist() 

        df = pd.concat([train_df, test_df], axis= 0)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # seperate test and valid data
        self.test_index = df.answerCode == -1
        self.valid_index = np.array(self.test_index[1:].tolist() + [False])

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        
        return df

    def load_train_data(self, test_file_name, train_file_name):
        data = self.load_data_from_file(test_file_name, train_file_name)
        # train data에는 test data의 유저에 대한 정보가 포함되면 안되므로(양심상, 규제는 완화된 것으로 보이기때문에 이 부분은 따로 수정해줄 필요도 있을 것 같다)
        self.train_data = data.merge(pd.Series(self.train_userID, name='userID'), how = 'inner', on = 'userID')
        self.valid_data = data[self.valid_index].query("answerCode != -1")

    def load_test_data(self, test_file_name, train_file_name):
        data = self.load_data_from_file(test_file_name, train_file_name, is_train=False)
        self.test_data = data[self.test_index].copy()
        self.test_data = self.test_data.drop('answerCode', axis=1)
        self.test_data = self.test_data[self.FEATS]

    def convert_dataset(self, train_data, valid_data):
        if self.args.model == 'lightgbm':
            lgb_train, lgb_valid, X_valid, y_valid = self.get_lgb_data(train_data, valid_data, self.FEATS)
            return lgb_train, lgb_valid, X_valid, y_valid
        
        elif self.args.model == 'catboost':
            cb_train, cb_valid, X_valid, y_valid = self.get_cb_data(train_data, valid_data, self.FEATS, self.CATS)
            return cb_train, cb_valid, X_valid, y_valid


    def get_lgb_data(self, train, valid, FEATS):
        # X, y 값 분리
        y_train = train['answerCode']
        X_train = train.drop(['answerCode'], axis=1)

        y_valid = valid['answerCode']
        X_valid = valid.drop(['answerCode'], axis=1)

        # lgb_train = lgb.Dataset(X_train[FEATS], y_train)
        # lgb_valid = lgb.Dataset(X_valid[FEATS], y_valid)

        X_train = X_train[FEATS]    
        X_valid = X_valid[FEATS]

        return X_train, y_train, X_valid, y_valid

    def get_cb_data(self, train, valid, FEATS, cat_cols):
        y_train = train['answerCode']
        X_train = train.drop(['answerCode'], axis=1)

        y_valid = valid['answerCode']
        X_valid = valid.drop(['answerCode'], axis=1)

        cb_train = Pool(data=X_train[FEATS], label=y_train, cat_features=cat_cols)
        cb_valid = Pool(data=X_valid[FEATS], label=y_valid, cat_features=cat_cols)

        X_valid = X_valid[FEATS]

        return cb_train, cb_valid, X_valid, y_valid


  
