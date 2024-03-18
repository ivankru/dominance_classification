import os
import numpy as np
import torch
import lightning.pytorch as pl
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score, auroc
import torchvision.models.video as tvmv
import sklearn.metrics as skm
import click
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import Subset
import pickle
import time
import copy
import gc
from typing import List, Optional, Any, Tuple

import sys
sys.path.insert(1, './')
print('Updated sys.path:', sys.path)

from utils import parse_config, seed_everything
from mednext import MedNeXt
from dataset_class import HeartLR_simple_main

def get_studyid_nums(studies: np.array, studyid_ordernum_dict: dict) -> List[int]:
    """
    Function for obtaining frames corresponding to the patient from the dataset

    studies - list of considered patients
    studyid_ordernum_dict - dictionary, where key - patient number, value - list of corresponding frames
    
    return: indexes of the considered patients
    """
        
    nums_list = []

    for study in studies:
        if study in studyid_ordernum_dict.keys():
            nums_list.extend(studyid_ordernum_dict[study])

    assert len(nums_list) > 0
    return nums_list

class HeartLightningModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        video_shape: Tuple[int, int],
        lr: float = 3e-4,
        weight_decay: float = 0,
        weight_path: str = None,
        max_epochs: int = None,
        **kwargs,
    ) -> None:
        """
        Class for training a dominance or occlusion classification model

        num_classes - number of classes
        video_shape - video size
        lr - learning rate
        weight_decay - weight_decay
        weight_path - path to weights for the model
        max_epochs - number of epochs

        return: None
        """

        self.save_hyperparameters() 
        super().__init__()
        self.num_classes = num_classes

        # ResNet models
        # self.model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights)
        self.model = tvmv.r3d_18(weights=None)     
        
        # override the number of classes in the last layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        self.lr = lr
        self.loss_func = nn.BCEWithLogitsLoss()
        self.example_input_array = Tensor(1, *video_shape)


        # mednext model (if used, uncomment and comment the ResNet model)
        # self.model = MedNeXt(
        #     in_channels = 3, 
        #     n_channels = 32,
        #     n_classes = num_classes,
        #     exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
        #     # exp_r = 2,
        #     kernel_size=3,                     # Can test kernel_size
        #     deep_supervision=False,             # Can be used to test deep supervision
        #     do_res=True,                      # Can be used to individually test residual connection
        #     do_res_up_down = True,
        #     block_counts = [3,4,8,8,8,8,8,4,3],
        #     dim = '3d',
        #     grn=True    
        # )

        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path), strict=False)

        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        # lists with values obtained on validation
        self.y_val = []
        self.p_val = []
        self.r_val = []

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):

        # model training
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = y_hat

        # logging metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy(y_pred, y, task="binary"), prog_bar=True)
        self.log("train_f1", f1_score(y_pred, y, task="binary"), prog_bar=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        # model validation
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.sigmoid(y_hat) 

        # add the resulting values to the lists
        self.y_val.append(int(y[...,0].cpu()))
        self.p_val.append(float(y_pred[...,0].cpu()))
        self.r_val.append(round(float(y_pred[...,0].cpu())))

        return loss

    def on_validation_epoch_end(self) -> torch.Tensor:

        #logging and cleaning of lists at the end of the validation epoch
        try:
            self.log("val_roc_auc", skm.roc_auc_score(self.y_val, self.p_val), prog_bar=True)
            self.log("val_f1_score", skm.f1_score(self.y_val, self.r_val), prog_bar=True)
            self.log("val_accuracy", skm.accuracy_score(self.y_val, self.r_val), prog_bar=True)
        except ValueError as err: 
            print(err)
            print("Y_VAL", self.y_val)
            print("P_VAL", self.p_val)
            self.log("val_f1_score", skm.f1_score(self.y_val, self.r_val), prog_bar=True)

        self.y_val.clear()
        self.p_val.clear()
        self.r_val.clear()
        

    def on_train_epoch_end(self) -> None:

        # logging at the end of the training epoch
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Any:

        # optimizer parameters for model training
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]  
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        # getting a prediction
        x, y = batch
        y_hat = self(x)
        y_pred = torch.sigmoid(y_hat)
        return {"y": y, "y_pred": torch.round(y_pred), "y_prob": y_pred}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


# initialization of the parser as a decorator
@click.command()
@click.option("-en", "--exp-number", type=str, required=True, help="experiment number")
@click.option("-nc", "--num-classes", type=int, default=1, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=4, help="batch size.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--max-epochs", type=int, default=50, help="max epochs.")
@click.option("--num-workers", type=int, default=8)
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")
def main(
    exp_number: int,
    num_classes: int, 
    batch_size: int,
    video_size : Tuple[int, int],
    max_epochs: int,
    num_workers: int,
    fast_dev_run: bool,
    seed: int,
) -> None:
    """
    Function to run the script

    exp_number - number of experiment for the config
    num_classes - number of classes
    batch_size - batch size
    video_size - video resolution
    max_epochs - number of epochs
    num_workers - number of workers
    fast_dev_run - flag, whether accelerated format is needed
    seed - seed

    return: None
    """

    pl.seed_everything(seed)

    # optimal mean and sk for mean-std normalization obtained on ImageNet dataset
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # initialization of training and test transformations
    train_transform = T.Compose([
        ToTensorVideo(),  # C, T, H, W
        Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
        RandAugment(magnitude=10, num_layers=2),
        Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
        T.Resize(size=video_size, antialias=False),
        Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    test_transform = T.Compose([
        ToTensorVideo(),
        T.Resize(size=video_size, antialias=False),
        Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # the path to the config
    path = f"configs/exp_{exp_number}/train_cross_val_config.yaml"

    # setting parameters from config
    config = parse_config(path)

    # list of features under consideration
    features = config['features'].split(", ")

    # experiment number for the folder name
    exp_number = config['exp_number']

    # dataset folder
    dataset_folder = config['dataset_folder']

    # folder with a file with information on fold splits
    folds_studies_path = config['folds_studies_path']

    # device
    device = config['device']
    print(f"Device: {device}")

    # dictionary with dataset parameters
    dataset_params = config['dataset_params']

    # dictionary with slice selection parameter settings
    slice_selection_params = config['slice_selection_settings']

    # current label type
    labels_type = config['labels_type']
    print(f"labels type: {labels_type}")

    seed = config['seed']

    print("save path experiment number -", exp_number)
    print("artery type:", dataset_params["artery_type"])

    # path to the folder with models
    models_folder = os.path.join(config['models_folder'], f'exp_{exp_number}')

    # time delay to verify that the settings provided are correct
    time.sleep(10)

    # download with information about the breakdown into folds
    with open (folds_studies_path,'rb') as pick:
        folds = pickle.load(pick)["folds"]

    # fixing seed
    seed_everything(seed=seed)

    # creating a dataset
    dataset = HeartLR_simple_main(dataset_folder, features=features, transform=None,
                                slice_selection_params=slice_selection_params, device=device, 
                                labels_type=labels_type, **dataset_params) 

    dataset.transform = train_transform

    for i in range(len(folds['train'])):
        print()
        print(f'***** Fold {i} *****')
        print()

        # create lists for training and validation
        train_studies = folds['train'][i]
        val_studies = folds['valid'][i]
        test_studies = folds['test'][i]

        train_studies = np.copy(np.concatenate([train_studies, val_studies]))
        val_studies = np.copy(test_studies)

        train_set = Subset(copy.deepcopy(dataset), get_studyid_nums(train_studies, dataset.studyid_ordernum_dict))
        dataset.transform = test_transform
        val_set = Subset(dataset, get_studyid_nums(val_studies, dataset.studyid_ordernum_dict))

        print("Subset was created")

        train_dataloader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            val_set,
            batch_size=1, 
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

        print("DataLoader was initialized")

        # get the size of the video
        x, y = next(iter(train_dataloader))
        print(x.shape)

        # initialization of the class with the model
        model = HeartLightningModule(
            num_classes=num_classes,
            video_shape=x.shape[1:],
            lr=1e-4,
            weight_decay=0.001,
            max_epochs=max_epochs,
        )

        print("Model was initialized")

        # Initialization of learning stop when the metric stops growing
        early_stopping = EarlyStopping(
                            monitor="val_f1_score",
                            min_delta=0.001,
                            patience=5,
                            mode="max"
                        )
        
        # logger initialization
        logger = TensorBoardLogger(f"logs_tensorboard", name=f"exp_{exp_number}/fold_{i}")

        print("Logger is running")

        # Initialization of the model training tool
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            min_epochs=20,
            accelerator="gpu",
            fast_dev_run=fast_dev_run,
            logger=logger,
            callbacks=[early_stopping],
            log_every_n_steps=10,
            devices=-1
        )

        # clearing memory
        gc.collect()
        torch.cuda.empty_cache() 

        # model training
        print("Beginning model training")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{models_folder}/r3d_type_fold_{i}.pt")

        print()
        print('The end of learning the fold')
        print()


if __name__ == "__main__":
    main()