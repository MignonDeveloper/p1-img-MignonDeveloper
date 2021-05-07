import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import argparse
from PIL import Image
import os, random, math, pprint
from tqdm import tqdm
from importlib import import_module

import neptune.new as neptune

from mask_dataset import MaskTrainDataset, MaskValDataset
from loss import create_criterion
from optimizer import create_optimizer
from scheduler import create_scheduler
from pytorch_tools import EarlyStopping

# neptune.ai 를 활용한 experiment 관리
run = neptune.init(project='user_name/Project_name',
				   api_token='user_token',
                   source_files='*.py')


class CFG:
    PROJECT_PATH = "/opt/ml" # 기본 프로젝트 디렉터리
    BASE_DATA_PATH = '/opt/ml/input/data/train' # 데이터가 저장된 디렉터리

    learning_rate = 1e-3
    batch_size = 32
    num_workers = 4
    print_freq = 1
    nepochs = 40
    seed = 42
    patience = 10 # patience for Early stopping
    resize_width = 380
    resize_height = 380

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = "EfficientNetMaskClassifier"
    kfold = 0
    train_augmentation = "BaseAugmentation"
    test_augmentation = "BaseAugmentation"
    optimizer = "Adam"
    criterion = "cross_entropy"
    scheduler = "StepLR"
    description = "Modeling" 

    img_dir = 'images' # image directory path
    csv_path = 'train_st_df.csv' # train csv file
    docs_path = 'docs' # result, visualization image path
    model_path = 'models' # trained model path


# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="Mask Classification")

    # Container environment
    parser.add_argument('--PROJECT_PATH', type=str, default=CFG.PROJECT_PATH)
    parser.add_argument('--BASE_DATA_PATH', type=str, default=CFG.BASE_DATA_PATH)
    parser.add_argument('--csv_path', type=str, default=CFG.csv_path)

    # hyper parameters
    parser.add_argument("--lr", type=float, default=CFG.learning_rate, help=f'learning rate (defalut: {CFG.learning_rate})')
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help=f'input batch size for training (default: {CFG.batch_size})')
    parser.add_argument("--nworkers", type=int, default=CFG.num_workers, help=f'num workers for data loader (default: {CFG.num_workers})')
    parser.add_argument("--nepochs", type=int, default=CFG.nepochs, help=f'number of epochs to train (default: {CFG.nepochs})')
    parser.add_argument("--seed", type=int, default=CFG.seed, help=f'random seed (default: {CFG.seed})')
    parser.add_argument("--patience", type=int, default=CFG.patience, help=f'early stopping patience (default: {CFG.patience})')
    parser.add_argument("--resize_width", type=int, default=CFG.resize_width, help='resize_width size for image when training')
    parser.add_argument("--resize_height", type=int, default=CFG.resize_height, help='resize_height size for image when training')
    
    # network environment selection
    parser.add_argument('--model', type=str, default=CFG.model, help=f'model type (default: {CFG.model})')
    parser.add_argument('--kfold', type=int, default=CFG.kfold, help=f'K-Fold (default: {CFG.kfold})')
    parser.add_argument('--train_augmentation', type=str, default=CFG.train_augmentation, help=f'train data augmentation type (default: {CFG.train_augmentation})')
    parser.add_argument('--test_augmentation', type=str, default=CFG.test_augmentation, help=f'test data augmentation type (default: {CFG.test_augmentation})')
    parser.add_argument('--optimizer', type=str, default=CFG.optimizer, help=f'optimizer type (default: {CFG.optimizer})')
    parser.add_argument('--criterion', type=str, default=CFG.criterion, help=f'criterion type (default: {CFG.criterion})')
    parser.add_argument('--scheduler', type=str, default=CFG.scheduler, help=f'scheduler type (default: {CFG.scheduler})')
    parser.add_argument('--description', type=str, default=CFG.description, help='model description')

    args = parser.parse_args()
    # print(args) # for check arguments
    
    CFG.PROJECT_PATH = args.PROJECT_PATH
    CFG.BASE_DATA_PATH = args.BASE_DATA_PATH
    CFG.csv_path = args.csv_path

    CFG.learning_rate = args.lr
    CFG.batch_size = args.batch_size
    CFG.num_workers = args.nworkers
    CFG.nepochs = args.nepochs
    CFG.seed = args.seed
    CFG.patience = args.patience
    CFG.resize_width = args.resize_width
    CFG.resize_height = args.resize_height      

    CFG.model = args.model
    CFG.kfold = args.kfold
    CFG.train_augmentation = args.train_augmentation
    CFG.test_augmentation = args.test_augmentation
    CFG.optimizer = args.optimizer
    CFG.criterion = args.criterion
    CFG.scheduler = args.scheduler
    CFG.description = args.description

    CFG.img_dir = os.path.join(CFG.BASE_DATA_PATH, CFG.img_dir)
    CFG.csv_path = os.path.join(CFG.BASE_DATA_PATH, CFG.csv_path)
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, CFG.docs_path)
    CFG.model_path = os.path.join(CFG.PROJECT_PATH, CFG.model_path)
    
    # for check CFG
    # pprint.pprint(CFG.__dict__) 


def set_random_seed():
    # for Reproducible Model
    torch.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed) # if use multi-GPU


def set_logging():
    # experiments 관리 with  neptune.ai
    params = {
        "csv_path": CFG.csv_path,
        "learning_rate": CFG.learning_rate,
        "batch_size": CFG.batch_size,
        "nworkers": CFG.num_workers,
        "nepochs": CFG.nepochs,
        "random_seed": CFG.seed,
        "patience": CFG.patience,
        "resize_width": CFG.resize_width,
        "resize_height": CFG.resize_height,

        "model": CFG.model,
        "kfold": CFG.kfold,
        "train_augmentation": CFG.train_augmentation,
        "test_augmentation": CFG.test_augmentation,
        "optimizer": CFG.optimizer,
        "criterion": CFG.criterion,
        "scheduler": CFG.scheduler,
    }
    run["param"] = params
    run["description"] = CFG.description


def get_data():
    # read csv file containing image information
    train_df = pd.read_csv(CFG.csv_path)
    return train_df


def data_visualization(train_df):
    # visulization image along with information (mask formation, age, gender)
    choices = random.choices(range(len(train_df['path'])), k=6)
    img_paths = [ os.path.join(CFG.img_dir, train_df['path'][i]) for i in choices ]
    img_label = [ f"{train_df['path'][i].split('/')[-1]}, age: {train_df['age'][i]}, gender: {train_df['gender'][i]}" for i in choices ]
    imgs = [ Image.open(img_path).convert('RGB') for img_path in img_paths ]
    imgs_np = [ np.array(img) for img in imgs ]

    n_rows, n_cols = 2, 3
    _, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15, 12))
    for i in range(n_rows * n_cols):
        axes[i//(n_rows+1)][i%n_cols].imshow(imgs_np[i])
        axes[i//(n_rows+1)][i%n_cols].set_title(img_label[i], color='r')

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.docs_path, 'mask_train_data_viz.png'))


# train datset과 validation dataset에 사람이 겹쳐서 data leakage가 발생하지 않도록 stratified split
def get_stratified_data(train_df):
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

    X = train_df['path'].to_numpy()
    X = CFG.img_dir + '/' + X
    # y = train_df['gender_class', 'age_class', 'mask_class'].to_numpy()
    y = train_df['st_class'].to_numpy()

    for idx, (train_index, test_index) in enumerate(stratified_k_fold.split(X, y)):
        if idx == CFG.kfold:    
            X_stratified_train = X[train_index]
            X_stratified_test = X[test_index]
            y_stratified_train = y[train_index]
            y_stratified_test = y[test_index]
            break

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for idx, path in enumerate(X_stratified_train):
        filelist = os.listdir(path)
        
        for file in filelist:
            if file[0] == 'm':
                X_train.append(os.path.join(path, file))
                y_train.append(y_stratified_train[idx])
            
            elif file[0] == 'i':
                X_train.append(os.path.join(path, file))
                y_train.append(y_stratified_train[idx] + 6)

            elif file[0] == 'n':
                X_train.append(os.path.join(path, file))
                y_train.append(y_stratified_train[idx] + 12)

    for idx, path in enumerate(X_stratified_test):
        filelist = os.listdir(path)
        
        for file in filelist:
            if file[0] == 'm':
                X_test.append(os.path.join(path, file))
                y_test.append(y_stratified_test[idx])
            
            elif file[0] == 'i':
                X_test.append(os.path.join(path, file))
                y_test.append(y_stratified_test[idx] + 6)

            elif file[0] == 'n':
                X_test.append(os.path.join(path, file))
                y_test.append(y_stratified_test[idx] + 12)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def get_data_iter(X_stratified_train, X_stratified_test, y_stratified_train, y_stratified_test):

    train_transformer_module = getattr(import_module("augmentation"), CFG.train_augmentation)
    test_transformer_module = getattr(import_module("augmentation"), CFG.test_augmentation)

    train_transformer = train_transformer_module(resize_height=CFG.resize_height, resize_width=CFG.resize_width)
    test_transformer = test_transformer_module(resize_height=CFG.resize_height, resize_width=CFG.resize_width)

    mask_train_dataset = MaskTrainDataset(image_path=X_stratified_train, output_class=y_stratified_train, transformer=train_transformer)
    mask_test_dataset = MaskValDataset(image_path=X_stratified_test, output_class=y_stratified_test, transformer=test_transformer)

    train_iter = torch.utils.data.DataLoader(mask_train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    test_iter = torch.utils.data.DataLoader(mask_test_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)

    return mask_train_dataset, mask_test_dataset, train_iter, test_iter


def get_model(train_iter):
    # mask_model.py에 정의된 특정 모델을 가져옵니다.
    model_module = getattr(import_module("mask_model"), CFG.model)
    model = model_module()

    # 모델의 파라미터를 GPU 메모리로 옮깁니다.
    model.cuda()    
    
    # 모델의 파라미터 수를 출력합니다.
    print('parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # GPU가 2개 이상이면 데이터패러럴로 학습 가능하게 만듭니다.
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # loss.py에 정의된 criterion을 가져옵니다. 
    criterion_mask = create_criterion(CFG.criterion, classes=3, smoothing=0.05)
    criterion_gender = create_criterion('cross_entropy')
    criterion_age = create_criterion(CFG.criterion, classes=3, smoothing=0.05)

    # optimizer.py에 정의된 optimizer를 가져옵니다.
    optimizer_backbone = create_optimizer(
        CFG.optimizer,
        params=model.backbone.parameters(),
        lr = CFG.learning_rate * 0.1,
        momentum=0.9,
        weight_decay=1e-2
    )
    optimizer_classifier = create_optimizer(
        CFG.optimizer,
        params=[
            {"params": model.mask_layer.parameters()},
            {"params": model.gender_layer.parameters()},
            {"params": model.age_layer.parameters()},
        ],
        lr = CFG.learning_rate,
        momentum=0.9,
        weight_decay=1e-2
    )

    # scheduler.py에 정의된 scheduler를 가져옵니다.
    scheduler_backbone = create_scheduler(
        CFG.scheduler,
        optimizer=optimizer_backbone,
        max_lr=CFG.learning_rate * 0.1,
        epochs=CFG.nepochs,
        steps_per_epoch=len(train_iter),
        pct_start=5/CFG.nepochs,
        anneal_strategy='cos'
    )
    scheduler_classifier = create_scheduler(
        CFG.scheduler,
        optimizer=optimizer_classifier,
        max_lr=CFG.learning_rate,
        epochs=CFG.nepochs,
        steps_per_epoch=len(train_iter),
        pct_start=5/CFG.nepochs,
        anneal_strategy='cos'
    )

    return model, criterion_mask, criterion_gender, criterion_age, optimizer_backbone, optimizer_classifier, scheduler_backbone, scheduler_classifier


# evaluation function for validation data
def func_eval(model, criterion_mask, criterion_gender, criterion_age, test_dataset, test_iter):
    loss_total_sum = 0
    label = []
    predict = []

    model.eval() # make model evaluation mode
    with torch.no_grad():
        for batch_in, batch_mask, batch_gender, batch_age in test_iter:
        
            y_pred = model.forward(batch_in.to(CFG.device))
            loss_1 = criterion_mask(y_pred[0], batch_mask.to(CFG.device))
            loss_2 = criterion_gender(y_pred[1], batch_gender.to(CFG.device))
            loss_3 = criterion_age(y_pred[2], batch_age.to(CFG.device))
            loss_total_sum += (loss_1 + loss_2 + loss_3) * CFG.batch_size

            for idx in range(len(y_pred[0])):
                y_pred_mask = torch.max(y_pred[0][idx], dim = 0)[1].cpu()
                y_pred_gender = torch.max(y_pred[1][idx], dim = 0)[1].cpu()
                y_pred_age = torch.max(y_pred[2][idx], dim = 0)[1].cpu()

                predict_output_class = 6 * y_pred_mask.item() + 3 * y_pred_gender.item() + y_pred_age.item()
                predict.append(predict_output_class)

                label_output_class = 6 * batch_mask[idx].item() + 3 * batch_gender[idx].item() + batch_age[idx].item()
                label.append(label_output_class)

    model.train() # back to train mode

    accuracy = accuracy_score(y_true=label, y_pred=predict)
    f1 = f1_score(y_true=label, y_pred=predict, average="macro")
    loss_out = loss_total_sum / len(test_dataset)
    return loss_out, accuracy, f1


def train(model, criterion_mask, criterion_gender, criterion_age, optimizer_backbone, optimizer_classifier, scheduler_backbone, scheduler_classifier, train_dataset, test_dataset, train_iter, test_iter):
    print ("Start training.\n")
    model.train() # to train mode

    early_stopping = EarlyStopping(patience = CFG.patience, delta=0) # early stopping initializing
    scaler = GradScaler()

    for epoch in tqdm(range(CFG.nepochs)):
        train_loss_sum = 0

        for batch_in, batch_mask, batch_gender, batch_age in train_iter:
            # reset gradient 
            optimizer_backbone.zero_grad()   
            optimizer_classifier.zero_grad() 

            with autocast():
                # Forward path with mixed precision
                y_pred = model.forward(batch_in.to(CFG.device))
                loss_1 = criterion_mask(y_pred[0], batch_mask.to(CFG.device))
                loss_2 = criterion_gender(y_pred[1], batch_gender.to(CFG.device))
                loss_3 = criterion_age(y_pred[2], batch_age.to(CFG.device))
                loss_final = loss_1 + loss_2 + loss_3

            # Update
            scaler.scale(loss_final).backward()            # backpropagate
            scaler.step(optimizer_backbone)                # optimizer_backbone update
            scaler.step(optimizer_classifier)              # optimizer_classifier update
            scaler.update()                                # Updates the scale for next iteration.

            # scheduler step
            scheduler_backbone.step()                 
            scheduler_classifier.step()     

            train_loss_sum += loss_final.item() * CFG.batch_size


        # caculate train_loss
        train_loss = train_loss_sum / len(train_dataset)

        # Print
        if ((epoch % CFG.print_freq)==0) or (epoch==(CFG.nepochs - 1)):
            val_loss, accuracy, f1 = func_eval(model, criterion_mask, criterion_gender, criterion_age, test_dataset, test_iter)
            print ("epoch:[%d] train_loss:[%.5f] val_loss:[%.5f] val_accuracy:[%.5f] val_f1_score:[%.5f]" % (epoch, train_loss, val_loss, accuracy, f1))

        run["epoch/accuracy"].log(accuracy)
        run["epoch/train_loss"].log(train_loss)
        run["epoch/val_loss"].log(val_loss)
        run["epoch/f1_score"].log(f1)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        else:
            if epoch >= 7:
                torch.save(model.state_dict(), os.path.join(CFG.model_path, f'epoch_{epoch}.pt'))

    print ("Done")


def main():
    # check pytorch version & whether using cuda or not
    print ("PyTorch version:[%s]."%(torch.__version__))
    print ("device:[%s]."%(CFG.device))

    get_config()
    set_random_seed()
    set_logging()
    train_df = get_data()
    data_visualization(train_df)
    X_stratified_train, X_stratified_test, y_stratified_train, y_stratified_test = get_stratified_data(train_df)
    train_dataset, test_dataset, train_iter, test_iter = get_data_iter(X_stratified_train, X_stratified_test, y_stratified_train, y_stratified_test)

    model, criterion_mask, criterion_gender, criterion_age, optimizer_backbone, optimizer_classifier, scheduler_backbone, scheduler_classifier = get_model(train_iter)
    train(model, criterion_mask, criterion_gender, criterion_age, optimizer_backbone, optimizer_classifier, scheduler_backbone, scheduler_classifier, train_dataset, test_dataset, train_iter, test_iter)


if __name__ == "__main__":
    main()