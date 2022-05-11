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
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", 'testId_first','problem_mid_cat']

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

        df['normal_elapsed'] = df['elapsed'].apply(normalize_outlier)
        elapsed_test = df.groupby(['userID','testId']).mean()['normal_elapsed']
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
        correct_a.columns = ["accessment_mean", 'accessment_sum']

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
        ## FE. 9 : 이동 평균을 사용해 최근 n개 문제 평균 풀이 시간
        # 기본값은 3, 후에 갯수 조정을해도 좋을듯함
        # 3개의 값이 되지 않는 경우 0으로 처리
        df['mean_time_second'] = df.groupby(['userID'])['normal_elapsed'].rolling(3).mean().fillna(0).values
        ####################################

        ###################################
        # FE. 10 : 문항번호 + 태그 => 약간의 성능 하락
        df['problem_num'] = df['assessmentItemID'].str[-3:]
        df['access_tag'] = df['problem_num'].map(str) + '_' + df['KnowledgeTag'].map(str)
        ###################################

        ###################################
        # FE. 11 : 시험지 대분류 => 난이도 => 리더보드에서는 성능 하락
        df['testId_first'] = df['assessmentItemID'].str[1:4].map(str)
        ###################################

        ###################################
        # FE. 12 : 'userID', 'KnowledgeTag' 그룹화 => cumcount => 성능 약간 하락
        df['user_tag_cnt'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()
        ###################################

        ###################################
        # FE. 13 : tag 정답률 * accessmentItemID 정답률 => 성능 상승
        df['tag_accessmentID_mul'] = df['tag_mean'] * df['accessment_mean']
        ###################################

        ###################################
        # FE. 14 : 문제별 앞에 문제인지 뒤에문제인지 problem_mid_cat (catagory)
        df['problem_num'] = df['assessmentItemID'].str[-3:]

        testId_group = df.loc[:, ['testId', 'problem_num']]
        testId_group = testId_group.drop_duplicates()
        testId_group = testId_group.groupby('testId').agg('count').reset_index()
        testId_group.columns = ['testId', 'problem_cnt']

        testId_group['problem_cnt_mid'] = testId_group['problem_cnt'].apply(lambda x : x//2)
        df = pd.merge(df, testId_group, on=['testId'], how="left")
        df['problem_num'] = df['problem_num'].apply(lambda x : int(x))
        df['problem_mid_cat']  = df.apply(lambda x : 0 if int(x.problem_num) < x.problem_cnt_mid else 1, axis = 1)
        ###################################

        # (2) FEATS는 FE가 직접적으로 작동이 되는 부분에서 언급되는것이 좋을것 같다.
        self.FEATS = ['KnowledgeTag', 'assessmentItemID', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum',
            'elapsed','main_ca_correct_answer','main_ca_total_answer','main_ca_acc','elapsed_test',
            'accessment_mean', 'accessment_sum', 'ka_accessment_mean', 'ka_accessment_sum', 
            'mean_time_second', 'tag_accessmentID_mul','problem_mid_cat']
            # 'userID', 'user_tag_cnt']


        # TODO catboost는 Categorical columns name을 지정해줘야한다.
        #self.CATS = ['KnowledgeTag']
        self.CATS = ['KnowledgeTag', 'assessmentItemID','problem_mid_cat']

        df.sort_values(by=['userID','Timestamp'], inplace=True)
        return df

    def load_data_from_file(self, test_file_name, train_file_name=None, is_train = True):
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
            }          
        test_csv_file_path = os.path.join(self.args.data_dir, test_file_name)
        test_df = pd.read_csv(test_csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
        
        train_csv_file_path = os.path.join(self.args.data_dir, train_file_name)
        train_df = pd.read_csv(train_csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
        self.train_userID = train_df['userID'].unique().tolist() 

        df = pd.concat([train_df, test_df], axis= 0)
        df = self.__feature_engineering(df) # FEATS 결정
        df = self.__preprocessing(df, is_train) # 인코딩/타입 결정

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
        # self.train_data = data[self.valid_index == False].query("answerCode != -1")
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


  
