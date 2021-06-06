import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from pprint import pprint


class EfficientNetMaskClassifier(nn.Module):
    '''
        Backbone: EfficientNet-b4
        Classifier: After get feature map from backbone, make 3 heads to decision each feature
            - mask
            - gender
            - age
        
        forward output
            - mask  : [batch_size, 3]
            - gender: [batch_size, 2]
            - age   : [batch_size, 3]
    '''
    def __init__(self):
        super(EfficientNetMaskClassifier, self).__init__()

        # get number of backbone feature map feature
        backbone_pre = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.in_features = backbone_pre.classifier.in_features

        # set backbone except original classifier
        backbone = torch.nn.Sequential(*(list(backbone_pre.children())[:-1]))
        self.backbone = backbone

        # set 3 heads for mask, gender, age
        self.mask_layer = self.__get_classifier(3)
        self.gender_layer = self.__get_classifier(2)
        self.age_layer = self.__get_classifier(3)

        # initialize weights with kaiming_normalization in classifier heads
        self.classifier_layers = [self.mask_layer, self.gender_layer, self.age_layer]
        for layer in self.classifier_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear): # lnit dense
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    
    def forward(self, x):
        # get feature map from backbone
        backbone_output = self.backbone(x)

        # forward to 3 heads
        mask_class = self.mask_layer(backbone_output)
        gender_class = self.gender_layer(backbone_output)
        age_class = self.age_layer(backbone_output)

        return mask_class, gender_class, age_class


    def __get_classifier(self, output_features):
        """
        get linear layer for each head with different structure

        Args:
            output_features (int): number of output target for each head

        Returns:
            classifier (nn.Linear): linear layer for each classification head
        """        
        classifier = nn.Sequential(
            nn.Linear(self.in_features, output_features)
        )
        return classifier



class ResNestMaskClassifier(nn.Module):
    '''
        Backbone: resnest-200e
        Classifier: After get feature map from backbone, make 3 heads to decision each feature
            - mask
            - gender
            - age
        
        forward output
            - mask  : [batch_size, 3]
            - gender: [batch_size, 2]
            - age   : [batch_size, 3]
    '''
    def __init__(self):
        super(ResNestMaskClassifier, self).__init__()

        # get number of backbone feature map feature
        backbone_pre = timm.create_model('resnest200e', pretrained=True)
        self.in_features = backbone_pre.fc.in_features

        # set backbone except original classifier
        backbone = torch.nn.Sequential(*(list(backbone_pre.children())[:-1]))
        self.backbone = backbone

        # set 3heads for mask, gender, age
        self.mask_layer = self.get_classifier(3)
        self.gender_layer = self.get_classifier(2)
        self.age_layer = self.get_classifier(3)

        # initialize weights with kaiming_normalization in classifier heads
        self.classifier_layers = [self.mask_layer, self.gender_layer, self.age_layer]
        for layer in self.classifier_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear): # lnit dense
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    
    def forward(self, x):
        # get feature map from backbone
        backbone_output = self.backbone(x)

        # forward to 3 heads
        mask_class = self.mask_layer(backbone_output)
        gender_class = self.gender_layer(backbone_output)
        age_class = self.age_layer(backbone_output)

        return mask_class, gender_class, age_class


    def get_classifier(self, output_features):
        classifier = nn.Sequential(
            nn.Linear(self.in_features, output_features)
        )
        return classifier


# final classification module for Deti backbone
class DetiFinalClassifier(nn.Module):
    def __init__(self, in_features):
        super(DetiFinalClassifier, self).__init__()

        self.backbone_in_features = in_features

        self.mask_layer = nn.Linear(self.backbone_in_features, 3)
        self.gender_layer = nn.Linear(self.backbone_in_features, 2)
        self.age_layer = nn.Linear(self.backbone_in_features, 3)

        self.classifier_layers = [self.mask_layer, self.gender_layer, self.age_layer]
        for layer in self.classifier_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear): # lnit dense
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)
    

    def forward(self, x):
        mask_class = self.mask_layer(x)
        gender_class = self.gender_layer(x)
        age_class = self.age_layer(x)

        return mask_class, gender_class, age_class


class DeTiMaskClassifier(nn.Module):
    '''
        Backbone: vit_deit_base_patch16_384
        Classifier: After get feature map from backbone, make 3 heads to decision each feature
            - mask
            - gender
            - age
        
        forward output
            - mask  : [batch_size, 3]
            - gender: [batch_size, 2]
            - age   : [batch_size, 3]
    '''
    def __init__(self):
        super(DeTiMaskClassifier, self).__init__()
        self.net = timm.create_model('vit_deit_base_patch16_384', pretrained=True)
        self.net.head = DetiFinalClassifier(self.net.head.in_features)

    
    def forward(self, x):
        return self.net(x)


class EfficientNetDropoutMaskClassifier(nn.Module):
    '''
        Backbone: EfficientNet-b4 with dropout
        Classifier: After get feature map from backbone, make 3 heads to decision each feature
            - mask
            - gender
            - age
        
        forward output
            - mask  : [batch_size, 3]
            - gender: [batch_size, 2]
            - age   : [batch_size, 3]
    '''
    def __init__(self):
        super(EfficientNetDropoutMaskClassifier, self).__init__()

        # get number of backbone feature map feature
        backbone_pre = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.in_features = backbone_pre.classifier.in_features
        self.mid_featrues = int(self.in_features / 2)

        # set backbone except original classifier
        backbone = torch.nn.Sequential(*(list(backbone_pre.children())[:-1]))
        self.backbone = backbone

        # set 3 heads for mask, gender, age
        self.mask_layer = self.__get_classifier(3)
        self.gender_layer = self.__get_classifier(2)
        self.age_layer = self.__get_classifier(3)

        # initialize weights with kaiming_normalization in classifier heads
        self.classifier_layers = [self.mask_layer, self.gender_layer, self.age_layer]
        for layer in self.classifier_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear): # lnit dense
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    
    def forward(self, x):
        # get feature map from backbone
        backbone_output = self.backbone(x)

        # forward to 3 heads
        mask_class = self.mask_layer(backbone_output)
        gender_class = self.gender_layer(backbone_output)
        age_class = self.age_layer(backbone_output)

        return mask_class, gender_class, age_class


    def __get_classifier(self, output_features):
        """
        get linear layer for each head with different structure

        Args:
            output_features (int): number of output target for each head

        Returns:
            classifier (nn.Sequential): linear layer with batch normalization & dropout for each classification head
        """ 

        classifier = nn.Sequential(
            nn.Linear(self.in_features, self.mid_featrues),
            nn.BatchNorm1d(self.mid_featrues),
            nn.Dropout(0.7),
            nn.Linear(self.mid_featrues, output_features)
        )
        return classifier


if __name__ == "__main__":
    # test model forward with simple example
    x = torch.randn(2, 3, 380, 380)
    model = EfficientNetMaskClassifier()
    print(model)
    output = model(x)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)