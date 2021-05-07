import numpy as np
import pandas as pd
import os

# 기본 프로젝트 디렉터리
PROJECT_PATH = "/opt/ml"

# 데이터가 저장된 디렉터리
BASE_DATA_PATH = '/opt/ml/input/data/train'

class CFG:
    img_dir = os.path.join(BASE_DATA_PATH, 'images')
    csv_path = os.path.join(BASE_DATA_PATH, 'train.csv') # train csv 파일


def make_new_train_df():
    train_df = pd.read_csv(CFG.csv_path)
    gender_age = train_df[['gender','age']]

    st_classes = []
    for idx in range(train_df.shape[0]):
        gender, age = gender_age.iloc[idx, :]

        # make 58,59 to 60 for minimizing data imbalance
        if gender == 'male':
            if age < 30:
                st_class = 0
            elif age > 30 and age <58:
                st_class = 1
            else:
                st_class = 2
        else:
            if age < 30:
                st_class = 3
            elif age > 30 and age <58:
                st_class = 4
            else:
                st_class = 5
        st_classes.append(st_class)

    # check distribution of stratified_class
    train_df.insert(loc=5, column="st_class", value=st_classes)
    print(train_df['st_class'].value_counts())

    # make a csv file for making dataset & training model
    train_df.to_csv(os.path.join(BASE_DATA_PATH, 'train_st_df_58.csv'))


if __name__ == '__main__':
    make_new_train_df()