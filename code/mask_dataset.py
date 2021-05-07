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
        if self.transformer:
            transform_image = self.transformer(image = image_np)
        transformed_image = transform_image['image']

        # mask -> 0(wear), 1(incorrect), 2(not wear)
        # gender -> 0(male), 1(female)
        # age -> 0(<30), 1(<60), 2(>=60)
        mask_class, gender_class, age_class = self.get_class(self.output_class[idx])
                        
        return transformed_image, mask_class, gender_class, age_class

    
    def get_class(self, output_class):
        mask_class = int(output_class / 6)
        gender_class = int( (output_class % 6) / 3 )
        age_class = int( (output_class % 6) % 3 )
        return mask_class, gender_class, age_class


################################ Validation Data Set ####################################

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
        if self.transformer:
            transform_image = self.transformer(image = image_np)
        transformed_image = transform_image['image']

        # mask -> 0(wear), 1(incorrect), 2(not wear)
        # gender -> 0(male), 1(female)
        # age -> 0(<30), 1(<60), 2(>=60)
        mask_class, gender_class, age_class = self.get_class(self.output_class[idx])
                        
        return transformed_image, mask_class, gender_class, age_class

    
    def get_class(self, output_class):
        mask_class = int(output_class / 6)
        gender_class = int( (output_class % 6) / 3 )
        age_class = int( (output_class % 6) % 3 )
        return mask_class, gender_class, age_class

################################ Test Data Set ####################################

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
        if self.transformer:
            transform_image = self.transformer(image = image_np)
        transformed_image = transform_image['image']
        return transformed_image


