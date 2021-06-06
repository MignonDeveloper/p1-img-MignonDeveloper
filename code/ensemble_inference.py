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

    # predict envirionment
    batch_size = 16
    num_workers = 4
    seed = 42
    resize_width = 380
    resize_height = 380

    # model environment
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_augmentation = "BaseAugmentation"

    # data route
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
    parser.add_argument("--resize_width", type=int, default=CFG.resize_width, help='resize_width size for image when inference')
    parser.add_argument("--resize_height", type=int, default=CFG.resize_height, help='resize_height size for image when inference')

    # model selection
    parser.add_argument('--model', type=str, default=CFG.model, help=f'test model type (default: {CFG.model})')
    parser.add_argument('--test_augmentation', type=str, default=CFG.test_augmentation, help=f'test data augmentation type (default: {CFG.test_augmentation})')
    args = parser.parse_args()
    # print(args) # for check arguments
    
    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.batch_size = args.batch_size
    CFG.num_workers = args.nworkers
    CFG.seed = args.seed    
    CFG.resize_width = args.resize_width
    CFG.resize_height = args.resize_height   
    CFG.test_augmentation = args.test_augmentation

    # path setting
    CFG.img_dir = os.path.join(CFG.BASE_DATA_PATH, 'images') # image directory 경로
    CFG.submission_path = os.path.join(CFG.BASE_DATA_PATH, 'info.csv') # train csv 파일

    # for check CFG
    # pprint.pprint(CFG.__dict__) # for check CFG


def set_random_seed():
    # Reproducible Model을 만들기 위해 Random Seed를 고정한다.
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

    test_transformer_module = getattr(import_module("augmentation"), CFG.test_augmentation)
    test_transformer = test_transformer_module(resize_height=CFG.resize_height, resize_width=CFG.resize_width)

    mask_test_dataset = MaskTestDataset(image_path=X, transformer=test_transformer)
    test_iter = torch.utils.data.DataLoader(mask_test_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle = False)

    return submission, test_iter


def sum_probability(test_iter, model_name, mask_prob, gender_prob, age_prob):   
    model_module = getattr(import_module("mask_model"), CFG.model)
    model = model_module()
    model.load_state_dict(torch.load(os.path.join(CFG.model_path, model_name)))

    model.cuda()
    model.eval()

    prev_count = 0

    for image in tqdm(test_iter):
        current_count = prev_count + len(image)
        with torch.no_grad():
            pred1, pred2, pred3 = model.forward(image.to(CFG.device))

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            pred1 = pred1.detach().cpu().numpy()
            pred2 = pred2.detach().cpu().numpy()
            pred3 = pred3.detach().cpu().numpy()
            
            mask_prob[prev_count:current_count] += pred1
            gender_prob[prev_count:current_count] += pred2
            age_prob[prev_count:current_count] += pred3

        prev_count = current_count

    return mask_prob, gender_prob, age_prob


def inference(test_iter, submission):
    mask_prob = np.zeros((submission.shape[0], 3))
    gender_prob = np.zeros((submission.shape[0], 2))
    age_prob = np.zeros((submission.shape[0], 3))

    mask_prob, gender_prob, age_prob = sum_probability(test_iter, "model_v1.pt", mask_prob, gender_prob, age_prob)
    mask_prob, gender_prob, age_prob = sum_probability(test_iter, "model_v2.pt", mask_prob, gender_prob, age_prob)
    mask_prob, gender_prob, age_prob = sum_probability(test_iter, "model_v3.pt", mask_prob, gender_prob, age_prob)
    mask_prob, gender_prob, age_prob = sum_probability(test_iter, "model_v4.pt", mask_prob, gender_prob, age_prob)
    mask_prob, gender_prob, age_prob = sum_probability(test_iter, "model_v5.pt", mask_prob, gender_prob, age_prob)

    for i in len(mask_prob):
        submission.iloc[i:1] = 6*np.argmax(mask_prob[i], axis=1) + 3*np.argmax(gender_prob, axis=1) + np.argmax(age_prob, axis=1))

    submission.to_csv(os.path.join(CFG.docs_path, 'result', f'submission_{CFG.model_version}_{CFG.model_epoch}.csv'), index=False)


def main():
    print ("PyTorch version:[%s]."%(torch.__version__))
    print ("device:[%s]."%(CFG.device))

    get_config()
    set_random_seed()
    submission, test_iter = get_data()
    inference(test_iter, submission)


if __name__ == "__main__":
    main()