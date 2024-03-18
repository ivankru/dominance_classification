import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import transforms
from albumentations.augmentations.transforms import ImageCompression 

def train_transform_generator():
    """
    Function for creating training augmentations

    return: transforms for the train dataset
    """
        
    small_crop_pad = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.85, 1), ratio=(0.9, 1.1))])
    rotate = A.Rotate(10, p=1)
    # brightness = transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0.2, hue=0, always_apply=False, p=1)
    noise_gauss = transforms.GaussNoise(var_limit=(5.0, 15.0), mean=0, per_channel=True, always_apply=False, p=1)
    noise_jpeg = ImageCompression(quality_lower=10, quality_upper=10, p=1)
    tr_transform = A.Compose([A.Resize(224, 224), 
                              A.OneOf([small_crop_pad,
                                    #    brightness, 
                                       noise_gauss,
                                       noise_jpeg,
                                       rotate], p=0.8), 
                              ToTensorV2()])

    return tr_transform

def val_transform_generator():
    """
    Function for creating test augmentations

    return: transforms for the test dataset
    """
        
    val_transform =  A.Compose([A.Resize(224, 224), 
                                ToTensorV2()])
    
    return val_transform