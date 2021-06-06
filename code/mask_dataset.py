import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

class MaskTrainDataset(Dataset):
    """ Mask Classification Train Datset """

    def __init__(self, image_path, output_class, transformer = None):
        self.image_path = image_path
        self.output_class = output_class
        self.transformer = transformer
    
    
    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        image_np = np.array(image)

        # albumentation transfomer를 이용해 image에 다양한 환경을 제공
        if self.transformer:
            transform_image = self.transformer(image = image_np)
        transformed_image = transform_image['image']

        # 기존 target number를 착용형태, 성별, 나이 별 class로 변환한다.
        mask_class, gender_class, age_class = self.__get_class(self.output_class[idx])
                        
        return transformed_image, mask_class, gender_class, age_class


    def __get_class(self, output_class):
        """
        0 ~ 17사이에 존재하는 class number를 각각의 기준별로 다시 정의
            - mask -> 0(wear), 1(incorrect), 2(not wear)
            - gender -> 0(male), 1(female)
            - age -> 0(<30), 1(<60), 2(>=60)

        Args:
            output_class (int): 기존 dataset에 존재하는 0~17사이에 정의된 class number

        Returns:
            mask_class, gender_class, age_class: 기존 class를 바탕으로 각 기준별 새로운 target으로 변환
        """        
        mask_class = int(output_class / 6)
        gender_class = int( (output_class % 6) / 3 )
        age_class = int( (output_class % 6) % 3 )
        return mask_class, gender_class, age_class


class MaskValDataset(Dataset):
    """ Mask Classification Train Datset """

    def __init__(self, image_path, output_class, transformer = None):
        self.image_path = image_path
        self.output_class = output_class
        self.transformer = transformer
    
    
    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        image_np = np.array(image)

        # albumentation transfomer를 이용해 image에 다양한 환경을 제공
        if self.transformer:
            transform_image = self.transformer(image = image_np)
        transformed_image = transform_image['image']

        # 기존 target number를 착용형태, 성별, 나이 별 class로 변환한다.
        mask_class, gender_class, age_class = self.__get_class(self.output_class[idx])
                        
        return transformed_image, mask_class, gender_class, age_class

    
    def __get_class(self, output_class):
        mask_class = int(output_class / 6)
        gender_class = int( (output_class % 6) / 3 )
        age_class = int( (output_class % 6) % 3 )
        return mask_class, gender_class, age_class


class MaskTestDataset(Dataset):
    """ Mask Classification Test Datset """

    def __init__(self, image_path, transformer = None):
        self.image_path = image_path
        self.transformer = transformer
    
    
    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        image_np = np.array(image)
        
        # albumentation transfomer를 이용해 image에 다양한 환경을 제공
        if self.transformer:
            transform_image = self.transformer(image = image_np)
        transformed_image = transform_image['image']
        
        return transformed_image


