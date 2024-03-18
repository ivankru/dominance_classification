import numpy as np
import pandas as pd
import os
import shutil
import pickle
from importlib.machinery import SourceFileLoader
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import torchvision
import time
import copy 
import click
from typing import List, Optional

import sys
sys.path.insert(1, './')
print('Updated sys.path:', sys.path)

from utils import *
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

def print_num_of_slices(projection_list: List[dict], phase_name: Optional[str] = None) -> None:
    """
    Function for counting and outputting the number of frames for left and right dominance

    projection_list - dataset with information about frames
    phase_name - training or test phase

    return: None
    """

    not_occlusion_frames_c = 0
    occlusion_frames_c = 0

    for elem in projection_list: 
        if elem['is_occlusion']: 
            occlusion_frames_c += 1

        elif not elem['is_occlusion']: 
            not_occlusion_frames_c += 1

    print(f'{phase_name}: \n occlusion frames = {occlusion_frames_c}')
    print(f'{phase_name}: \n not occlusion frames = {not_occlusion_frames_c}')


@click.command()
@click.option("-en", "--exp-number", type=str, required=True, help="experiment number")
@click.option("--need_augmentations", type=bool, is_flag=True, show_default=True, default=True,
                help="flag, do we need augmentations or not")
def main(
    exp_number: str,
    need_augmentations: bool
) -> None:

    # the path to the config
    path = f"configs/exp_{exp_number}/train_cross_val_config.yaml"

    # selection of a solution to the problem of classifying occlusions or artifacts
    label_name = "occlusion"

    print("configfile experiment number -", exp_number)
    print("augmentation:", need_augmentations)

    # setting parameters from config
    config = parse_config(path)

    # list of features under consideration
    features = config['features'].split(", ")

    # experiment number for the folder name
    exp_number = config['exp_number']

    # path to the folder with models
    models_folder = os.path.join(config['models_folder'], f'exp_{exp_number}')

    # loading transformations
    transforms_folder = config['transforms_folder']
    transforms_module = SourceFileLoader('transforms', transforms_folder).load_module()

    # dataset folder
    dataset_folder = config['dataset_folder']

    # folder with a file with information on fold splits
    folds_studies_path = config['folds_studies_path']

    # number of epochs
    epochs = config['n_epochs']

    # threshold for deciding how to classify a slice
    conf_thr = config['conf_thr']

    # device
    device = config['device']
    print(f"Device: {device}")

    # dictionary with dataset parameters
    dataset_params = config['dataset_params']

    # dictionary with optimizer settings
    optimizer_settings = config['optimizer_settings']

    # dictionary with slice selection parameter settings
    slice_selection_params = config['slice_selection_settings']

    # dictionary with loss function settings
    loss_fn_settings = config['loss_fn_settings']

    seed = config['seed']

    # dictionary with dataloader settings
    dataloaders_params = config['dataloaders_params']

    # type of model under consideration
    model_type = config['model_type'] 
    print(f"Current model type: {model_type}")

    # path to the logs folder
    log_path = config['log_path']

    print("save path experiment number -", exp_number)
    print("artery type:", dataset_params["artery_type"])

    # download with information about the breakdown into folds
    with open (folds_studies_path,'rb') as pick:
        folds = pickle.load(pick)["folds"]

    # time delay to verify that the settings provided are correct
    time.sleep(10)

    # create a folder with models
    if os.path.isdir(models_folder):
        shutil.rmtree(models_folder)
    os.makedirs(models_folder)

    # fixing the seed
    seed_everything(seed=seed)

    # transformations
    tr_transform  = transforms_module.train_transform_generator()
    val_transform  = transforms_module.val_transform_generator()

    # creating a dataset
    dataset = HeartLR_simple_main(dataset_folder, features=features, label_name=label_name,
                                transform=tr_transform, slice_selection_params=slice_selection_params, device=device,
                                **dataset_params,)

    # Number of frames with left and right dominants in the dataset
    count_label_0 = dataset.count_label_0
    count_label_1 = dataset.count_label_1

    max_count_label = np.max([count_label_0, count_label_1])

    print("How many occlusions:", count_label_0)
    print("How many normal:", count_label_1)

    # automatic initialization of model weights based on the number of frames with left and right dominant features
    loss_fn_settings["weight"] = [max_count_label/count_label_0, max_count_label/count_label_1]
    print("Current weight", loss_fn_settings["weight"])

    print(f"Number of folds that will be trained: {len(folds['train'])}")

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

        # statistics output
        print(f'Number of the patients in the training part: {len(train_studies)}')
        print(f'Number of the patients in the valid part: {len(val_studies)}')
        
        # create a model
        if model_type == 'convnext':
            model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 
                                                    stochastic_depth_prob=.1)
            model.classifier[2] = torch.nn.Linear(768, 2)
        elif model_type == 'swin':
            model = torchvision.models.swin_v2_t(weights=torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1)
            model.head = torch.nn.Linear(768, 2)

        model = model.to(device)
        
        optimizer = AdamW(model.parameters(), **optimizer_settings)
        
        # create datasets based on the resulting partitioning
        if need_augmentations:
            dataset.transform = tr_transform
        else:
            dataset.transform = val_transform
        train_dataset = Subset(copy.deepcopy(dataset), get_studyid_nums(train_studies, dataset.studyid_ordernum_dict))
        dataset.transform = val_transform
        val_dataset = Subset(dataset, get_studyid_nums(val_studies, dataset.studyid_ordernum_dict))
        test_dataset = Subset(dataset, get_studyid_nums(test_studies, dataset.studyid_ordernum_dict))
        
        train_dataloader = DataLoader(train_dataset, **dataloaders_params['train_dataloader_params'])
        val_dataloader = DataLoader(val_dataset, **dataloaders_params['val_dataloader_params'])
        test_dataloader = DataLoader(test_dataset, **dataloaders_params['test_dataloader_params'])

        train_projection_list = [dataset.projection_list[i] for i in train_dataset.indices]
        val_projection_list = [dataset.projection_list[i] for i in val_dataset.indices]
        test_projection_list = [dataset.projection_list[i] for i in test_dataset.indices]
        
        # statistic output
        print('Number of slices in the dataset:')
        print_num_of_slices(train_projection_list, "train")
        print_num_of_slices(val_projection_list, "val")

        # select the appropriate loss function
        if config['loss_fn_type'] == 'CEandRCE':
            loss_fn = CEandRCE(**loss_fn_settings, device=device)
            loss_fn_test = CEandRCE(**loss_fn_settings, device=device)

        elif config['loss_fn_type'] == 'NCEandRCE':
            loss_fn = NCEandRCE(**loss_fn_settings, device=device)
            loss_fn_test = NCEandRCE(**loss_fn_settings, device=device)

        elif config['loss_fn_type'] == 'CE':
            weight = loss_fn_settings['weight']
            weight = torch.tensor(weight).float()
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            loss_fn_test = torch.nn.CrossEntropyLoss(weight=weight)
            loss_fn = loss_fn.to(device)
            loss_fn_test = loss_fn_test.to(device)
        
        if config['loss_fn_type'] == 'CEandRCE_normal':
            loss_fn = CEandRCE_normal(**loss_fn_settings, device=device)
            loss_fn_test = CEandRCE_normal(**loss_fn_settings, device=device)

        elif config['loss_fn_type'] == 'NCEandRCE_normal':
            loss_fn = NCEandRCE_normal(**loss_fn_settings, device=device)
            loss_fn_test = NCEandRCE_normal(**loss_fn_settings, device=device)

        # model training
        model_trainer = trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, train_dataloader=train_dataloader, 
                        val_dataloader=val_dataloader, device=device, fold=i, conf_thr=conf_thr)
        
        model_trainer.train_n_epochs(epochs=epochs, models_folder=models_folder)

        print()
        print('The end of learning the fold')
        print()

        # a selection of the best models
        best_model_loss, _, _ = find_best_models(models_folder, i)
        
        # obtaining results and entering them into a table
        for best_model, criterion in zip([best_model_loss], ['loss']):
            print(f'Критерий: {criterion}')
            print(best_model) 
            print()
            
            model.load_state_dict(torch.load(os.path.join(models_folder, best_model)))
        
            avg_loss, auc, f1, y_true_bad, y_pred_bad, images_bad, file_paths_bad, y_true_unsure, y_pred_unsure, images_unsure, metrics_dict = eval_nn(model=model, 
                                                                                                                                                    device=device, 
                                                                                                                                                    test_dataloader=test_dataloader, 
                                                                                                                                                    loss_fn=loss_fn_test, 
                                                                                                                                                    conf_thr=conf_thr)
            
            if criterion == 'loss':
                if i == 0:
                    df = pd.DataFrame(data=metrics_dict)
                    df['fold'] = i
                elif i > 0:
                    temp_df = pd.DataFrame(data=metrics_dict)
                    temp_df['fold'] = i

                    df = pd.concat([df, temp_df], axis=0, ignore_index=True)

            print(f'LOSS {avg_loss}\nAUC {auc}\nF1 {f1}')
            print()
        
    # saving results
    log_folder_path = os.path.join(log_path, f'exp_{exp_number}')
    if os.path.isdir(log_folder_path):
        shutil.rmtree(log_folder_path)

    os.mkdir(log_folder_path)
    log_excel_path = os.path.join(log_folder_path, 'train_logs.xlsx')
    df.to_excel(log_excel_path)

if __name__ == "__main__":
    main()