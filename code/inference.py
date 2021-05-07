import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from importlib import import_module
import os, random, argparse
from tqdm import tqdm
from pprint import pprint

from mask_dataset import MaskTestDataset


class CFG:
    PROJECT_PATH = "/opt/ml" # 기본 프로젝트 디렉터리
    BASE_DATA_PATH = '/opt/ml/input/data/eval' # Test 데이터가 저장된 디렉터리

    batch_size = 32
    num_workers = 4
    seed = 42

    resize_width = 380
    resize_height = 380

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = "EfficientNetMaskClassifier"
    test_augmentation = "BaseAugmentation"
    model_version = "0330"
    model_epoch = "0"

    img_dir = 'images'
    submission_path = 'info.csv'
    docs_path = 'docs'
    model_path = 'models'



# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="Mask Classification")

    # Container environment
    parser.add_argument('--PROJECT_PATH', type=str, default=CFG.PROJECT_PATH)
    parser.add_argument('--BASE_DATA_PATH', type=str, default=CFG.BASE_DATA_PATH)

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--nworkers", type=int, default=CFG.num_workers)
    parser.add_argument("--seed", type=int, default=CFG.seed)

    # model selection
    parser.add_argument('--model', type=str, default=CFG.model, help=f'model type (default: {CFG.model})')
    parser.add_argument('--test_augmentation', type=str, default=CFG.test_augmentation, help=f'test data augmentation type (default: {CFG.test_augmentation})')
    parser.add_argument("--model_version", type=str, required = True)
    parser.add_argument("--model_epoch", type=int, required = True)
    args = parser.parse_args()
    # print(args) # for check arguments
    
    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.batch_size = args.batch_size
    CFG.num_workers = args.nworkers
    CFG.seed = args.seed        
    CFG.model = args.model
    CFG.test_augmentation = args.test_augmentation
    CFG.model_version = args.model_version
    CFG.model_epoch = args.model_epoch

    # path setting
    CFG.img_dir = os.path.join(CFG.BASE_DATA_PATH, 'images') # image directory 경로
    CFG.submission_path = os.path.join(CFG.BASE_DATA_PATH, 'info.csv') # train csv 파일
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, 'docs') # result, visualization 저장 경로
    CFG.model_path = os.path.join(CFG.PROJECT_PATH, 'models') # trained model 저장 경로

    # for check CFG
    # pprint.pprint(CFG.__dict__) # for check CFG


def set_random_seed():
    # for Reproducible Model
    torch.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed) # if use multi-GPU


def get_data():
    submission = pd.read_csv(CFG.submission_path)
    X = submission['ImageID'].to_numpy()
    X = CFG.img_dir + '/' + X

    # get albumentation transformer for test dataset from augmentation.py
    test_transformer_module = getattr(import_module("augmentation"), CFG.test_augmentation)
    test_transformer = test_transformer_module(resize_height=CFG.resize_height, resize_width=CFG.resize_width)

    # get train & valid dataset from mask_dataset.py
    mask_test_dataset = MaskTestDataset(image_path=X, transformer=test_transformer)

    # define data loader based on test dataset
    test_iter = torch.utils.data.DataLoader(mask_test_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle = False)

    return submission, test_iter


# get saved model for inference
def get_model():
    # 미리 저장된 model의 구조를 가지는 모델을 recylce_model.py에서 가져옵니다.
    model_module = getattr(import_module("mask_model"), CFG.model)
    model = model_module()

    # 미리 저장된 모델의 정보를 그대로 load
    model.load_state_dict(torch.load(os.path.join(CFG.model_path, CFG.model_version, f'epoch_{CFG.model_epoch}.pt')))
    model.cuda() # load on GPU Memory
    model.eval() # make model eval mode
    return model


# 3개의 head에서 나온 결과를 바탕으로 0~17 사이의 클래스로 mapping
def get_class_label(mask_class, gender_class, age_class):
    result = []
    current_pred = 0
    for i in range(len(mask_class)):
        current_pred = mask_class[i] * 6 + gender_class[i] * 3 + age_class[i]
        result.append(current_pred)
        
    return np.array(result)


# make predictions
def inference(model, test_iter, submission):
    prev_count = 0

    for image in tqdm(test_iter):
        current_count = prev_count + len(image)
        with torch.no_grad():
            pred1, pred2, pred3 = model.forward(image.to(CFG.device))
            pred1 = pred1.detach().cpu().numpy()
            pred2 = pred2.detach().cpu().numpy()
            pred3 = pred3.detach().cpu().numpy()
            submission.iloc[prev_count:current_count,1] = get_class_label(np.argmax(pred1, axis=1), np.argmax(pred2, axis=1), np.argmax(pred3, axis=1))

        prev_count = current_count

    submission.to_csv(os.path.join(CFG.docs_path, 'result', f'submission_{CFG.model_version}_{CFG.model_epoch}.csv'), index=False)


def main():
    print ("PyTorch version:[%s]."%(torch.__version__))
    print ("device:[%s]."%(CFG.device))

    get_config()
    set_random_seed()
    submission, test_iter = get_data()
    model = get_model()
    inference(model, test_iter, submission)


if __name__ == "__main__":
    main()