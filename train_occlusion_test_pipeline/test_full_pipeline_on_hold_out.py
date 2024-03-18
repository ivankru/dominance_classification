import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from importlib.machinery import SourceFileLoader
import torch
from torch.nn import Softmax
from torch.utils.data import DataLoader, Subset
import torchvision
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, f1_score, accuracy_score
from tqdm import tqdm
import copy
import click
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from typing import List, Tuple, Optional, Any

import sys
sys.path.insert(1, './')
print('Updated sys.path:', sys.path)

from utils import *
from dataset_class import HeartLR_simple_main

def get_pred_lists_for_table(models_weighted_pred_dict: dict, models_weighted_true_dict) -> Tuple[List[str],
                                                                                                  List[float],
                                                                                                  List[float],
                                                                                                  List[float],
                                                                                                  List[float],
                                                                                                  List[float]]:
    """
    Function to retrieve lists from dictionaries with results for patients

    models_weighted_pred_dict - dictionary, where key - patient, values - derived metrics calculated on predictions
    models_weighted_true_dict - dictionary, where key - patient, values - derived metrics calculated on ground truth
    
    return: list with the ground true and results for the patients
    """

    # lists for storing relevant values for the patient
    labels_list = []
    confidence = []
    std_list = []
    entropy_list = []
    true_list = []
    true_confidence = []

    # getting values from dictionaries for predictions and adding them to lists
    for val in models_weighted_pred_dict.values():
        labels_list.append(val[0])
        confidence.append(round(val[1], 3))
        std_list.append(round(val[2], 3))
        entropy_list.append(round(val[3], 3))

    # получение значений из словарей для ground truth и добавление их в списки
    for val in models_weighted_true_dict.values():
        true_list.append(val[0])
        true_confidence.append(round(val[1], 3))

    return labels_list, confidence, std_list, entropy_list, true_list, true_confidence

def get_best_models_list(models_folder: str) -> List[str]:
    """
    Function to get weighted prediction for series from frame-by-frame predictions

    series_preds_dict - dictionaries. where key is the series number, values are the class probabilities for each frame

    return: paths for the best models
    """

    best_models_list = []

    for i in range(5):
        best_model_loss_fold, _, _ = find_best_models(models_folder, i)
        best_models_list.append(best_model_loss_fold)

    return best_models_list

def get_weighted_series_pred(series_preds_dict: dict) -> dict:
    """
    Function to get weighted prediction for series from frame-by-frame predictions

    series_preds_dict - dictionaries. where key is the series number, values are the class probabilities for each frame

    return: dict with the classes probabilities for the series
    """

    # dictionary, where key is series number, values are averaged probabilities of classes
    series_weighted_preds_dict = {}
    
    for key in series_preds_dict.keys():

        # calculate average class probabilities for the series
        preds_array = torch.tensor(series_preds_dict[key])
        preds_mean = torch.mean(preds_array, axis=0)

        series_weighted_preds_dict[key] = preds_mean
    return series_weighted_preds_dict

def get_weighted_study_pred(series_weighted_preds_dict: dict, dataset_study_series_dict: dict) -> dict:
    """
    Function to derive a weighted prediction for patients from series predictions

    series_weighted_preds_dict - dictionary where key is series, values are average class probabilities
    dataset_study_series_dict - dictionary of matching patients and their series numbers

    return: dict with the classes probabilities for the studies
    """

    # dictionary, where key is patient number, values are averaged probabilities of classes
    study_weighted_preds_dict = {}
    
    for study in dataset_study_series_dict.keys():

        # list with class probabilities for the series that apply to current patient
        preds_list = []
        for series in dataset_study_series_dict[study]:
            if series not in series_weighted_preds_dict:
                continue
            preds_list.append(series_weighted_preds_dict[series])

        if preds_list == []:
            continue
        
        # calculate the average label, std and probability of the resulting prediction for the patient
        preds_array = torch.stack(preds_list)
        preds_mean = torch.mean(preds_array, axis=0)

        weighted_pred_label = torch.argmax(preds_mean).item()
        weighted_pred_std = torch.std(preds_array[:, weighted_pred_label]).item()
        labels = torch.argmax(preds_array, axis=1)
        _, counts = np.unique(labels, return_counts=True)
        entropy = (-counts/len(labels) * np.log2(counts/len(labels))).sum()
        weighted_pred = torch.max(preds_mean).item()

        study_weighted_preds_dict[study] = [weighted_pred_label, weighted_pred, weighted_pred_std, entropy]
    return study_weighted_preds_dict

def get_study_preds_for_model(models_folder: str, model: Any, dataloader: DataLoader, best_model_loss: str,
                              dataset_study_series_dict: dict, predicted_series_list: Optional[List[str]] = None,
                              model_type: Optional[str] = None, dataset_series_dict: Optional[dict] = None,
                              device: str = "cpu") -> Tuple[dict, dict]:
    """
    Function to get weighted prediction for patients from frame-by-frame predictions

    models_folder - path to the folder with models for loading scales
    model - model for prediction by frames without loaded weights
    dataloader - dataloader
    best_model_loss - best weights for model by loss function
    dataset_study_series_dict - dictionary, where the key is the patient number and the value is the corresponding series
    predicted_series_list - list of series to be predicted
    model_type - current model type
    dataset_series_dict - dictionary, where the key is the series number and the value is the corresponding frame numbers in the dataset.
    
    return: dicts with the ground true and predictions values for the patients
    """

    # counters for counting certain and uncertain predictions
    global certain_predicsion_count
    global uncertain_prediction_count

    # dictionaries with frame-by-frame prediction results for series
    series_preds_dict = {}
    series_true_dict = {}

    # load weights into the model
    model = copy.deepcopy(model)
    model.load_state_dict(torch.load(os.path.join(models_folder, best_model_loss), map_location=device))
    model.train(False)

    with torch.no_grad():
        for data in tqdm(dataloader):

            # load the labels corresponding to the current classification
            if model_type == "dominance":
                inputs, _, labels, seriesid = data
            elif model_type == "occlusion":
                inputs, labels, _, seriesid = data 
            else:
                raise AssertionError(f"Wrong model type: {model_type}")
            
            # make predictions for the frames
            inputs = inputs.to(device)

            outputs = model(inputs)

            # getting predictions for the convnextv2 model
            if str(model).lower().startswith('convnextv2forimageclassification'):
                outputs = outputs['logits']

            outputs = outputs.to(device)

            y_pred = Softmax(dim=1)(outputs)

            # count certain and uncertain predictions for statistics
            for prob in torch.max(y_pred, axis=1).values:
                if prob > 0.8 or prob < 0.2:
                    certain_predicsion_count += 1
                else:
                    uncertain_prediction_count += 1

            # add predictions for the batches to the corresponding dictionaries
            for i in range(len(seriesid)):

                # if predictions for the current series have not been previously obtained, then update the dictionary for ground truth and create a list for class probabilities by frame
                if str(dataset_series_dict[int(seriesid[i].item())]) not in series_preds_dict.keys():
                    list_true = [0.] * 2
                    list_true[int(labels[i].item())] = 1.
                    series_true_dict[str(dataset_series_dict[int(seriesid[i].item())])] = [list_true]
                    if predicted_series_list is not None and str(dataset_series_dict[int(seriesid[i].item())]) not in predicted_series_list:
                        continue
                    series_preds_dict[str(dataset_series_dict[int(seriesid[i].item())])] = [list(y_pred[i].cpu().detach().numpy())]
                
                # if predictions for the current series have already been obtained for some frame, then add the current predictions to this list
                else:
                    list_true = [0.] * 2
                    list_true[int(labels[i].item())] = 1.
                    series_true_dict[str(dataset_series_dict[int(seriesid[i].item())])] = [list_true]
                    series_preds_dict[str(dataset_series_dict[int(seriesid[i].item())])].append(list(y_pred[i].cpu().detach().numpy()))

    # averaging the predictions for the series
    series_weighted_preds_dict =  get_weighted_series_pred(series_preds_dict)
    series_weighted_true_dict =  get_weighted_series_pred(series_true_dict)

    print(len(series_weighted_preds_dict), len(series_weighted_true_dict))

    # averaging predictions for patients
    study_weighted_preds_dict = get_weighted_study_pred(series_weighted_preds_dict, dataset_study_series_dict)
    study_weighted_true_dict = get_weighted_study_pred(series_weighted_true_dict, dataset_study_series_dict)

    print(len(study_weighted_preds_dict), len(study_weighted_true_dict))

    return study_weighted_preds_dict, study_weighted_true_dict

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

# counters for certain and uncertain predictions to calculate statistics
uncertain_prediction_count = 0
certain_predicsion_count = 0

@click.command()
@click.option("-en", "--exp-number", type=str, required=True, help="experiment number")
@click.option("--n_first_frames_if_empty", type=bool, is_flag=True, show_default=True, default=False,
                help="flag, do we need n_first_frames method if there are not good frames")
@click.option("--ssim_if_empty", type=bool, is_flag=True, show_default=True, default=False,
                help="flag, do we need ssim method if there are not good frames")
def main(
    exp_number: str,
    n_first_frames_if_empty: bool, 
    ssim_if_empty: bool
) -> None:

    print(f"Empty method n_first_frames: {n_first_frames_if_empty}")
    print(f"Empty method ssim: {ssim_if_empty}")

    # path to the config folder
    path = f"configs/exp_{exp_number}/train_cross_val_config.yaml"

    # setting parameters from config
    config = parse_config(path)

    # list of features under consideration
    features = config['features'].split(", ")

    # number of classes to solve the classification problem
    num_classes = config["num_classes"]

    # experiment number for the folder name
    exp_number = config['exp_number']

    # path to the folder with models for solving the dominance classification problem for the right artery type
    models_folder_dominance_rca = config['models_folder_dominance_rca']

    # path to the folder with models for solving the dominance classification problem for the left artery type
    models_folder_dominance_lca = config['models_folder_dominance_lca']

    # path to the folder with models for solving the occlusion classification problem for the right artery type
    models_folder_occlusion = config['models_folder_occlusion']

    # loading transformations
    transforms_folder = config['transforms_folder']
    transforms_module = SourceFileLoader('transforms', transforms_folder).load_module()

    # dataset folder
    dataset_folder = config['dataset_folder']

    # device
    device = config['device']
    print(f"Device: {device}")

    # dictionary with dataset parameters
    dataset_params = config['dataset_params']

    # folder with a file with information on fold splits
    folds_studies_path = config['folds_studies_path']

    # dictionaries with settings of slice selection parameters for right and left artery types
    test_slice_selection_params_rca = config['test_slice_selection_settings_rca']
    test_slice_selection_params_lca = config['test_slice_selection_settings_lca']

    seed = config['seed']

    # dictionary with dataloader settings
    dataloaders_params = config['dataloaders_params']

    # model type for occlusion classification
    model_type_occlusion = config['model_type_occlusion']

    # type of model for dominance classification
    model_type_dominance = config['model_type_dominance']

    print(f"Current model type occlusion: {model_type_occlusion}")
    print(f"Current model type dominance: {model_type_dominance}")

    # path to the logs folder
    log_path = config['log_path']

    print("save path experiment number -", exp_number)

    # fixing seed
    seed_everything(seed=seed)

    # transforms
    val_transform  = transforms_module.val_transform_generator()

    # initialization of the model to solve the dominance classification problem
    if model_type_dominance == 'convnext':
        model_dominance = torchvision.models.convnext_tiny(weights=None, num_classes=2)
    elif model_type_dominance == 'swin':
        model_dominance = torchvision.models.swin_v2_t(weights=None, num_classes=2)
    elif model_type_dominance == 'convnext_v2':
        configuration = ConvNextV2Config()
        model_dominance = ConvNextV2ForImageClassification(configuration)

    # initialization of the model to solve the occlusion classification problem
    if model_type_occlusion == 'convnext':
        model_occlusion = torchvision.models.convnext_tiny(weights=None, num_classes=2)
    elif model_type_occlusion == 'swin':
        model_occlusion = torchvision.models.swin_v2_t(weights=None, num_classes=2)
    elif model_type_occlusion == 'convnext_v2':
        configuration = ConvNextV2Config()
        model_dominance = ConvNextV2ForImageClassification(configuration)

    model_dominance = model_dominance.to(device)
    model_occlusion = model_occlusion.to(device)

    # load patient partitioning and hold out information
    with open (folds_studies_path,'rb') as pick:
        folds_and_holdout_dict = pickle.load(pick)
        folds = folds_and_holdout_dict["folds"]
        holdout = folds_and_holdout_dict["holdout"]

    print("RCA dataset is creating...")

    # creating a dataset for the right artery type
    dataset_rca = HeartLR_simple_main(dataset_folder, features=features, status="test", returns_type="pipeline",
                            transform=val_transform, slice_selection_params=test_slice_selection_params_rca, device=device, artery_type="RCA",
                            **dataset_params, n_first_frames_if_empty=n_first_frames_if_empty, ssim_if_empty=ssim_if_empty, holdout_studies=holdout)
    dataloader_rca = DataLoader(dataset_rca, **dataloaders_params['test_dataloader_params'])

    print("LCA dataset is creating...")

    # creating a dataset for the left artery type
    dataset_lca = HeartLR_simple_main(dataset_folder, features=features, status="test", returns_type="pipeline",
                            transform=val_transform, slice_selection_params=test_slice_selection_params_lca, device=device, artery_type="LCA",
                            **dataset_params, n_first_frames_if_empty=n_first_frames_if_empty, ssim_if_empty=ssim_if_empty, holdout_studies=holdout)
    dataloader_lca = DataLoader(dataset_lca, **dataloaders_params['test_dataloader_params'])

    print("RCA dataset size=", len(dataset_rca))
    print("LCA dataset size=", len(dataset_lca))

    # finding a list of best-loss models to solve all problems for classification
    best_models_dominance_rca_list = get_best_models_list(models_folder_dominance_rca)
    best_models_dominance_lca_list = get_best_models_list(models_folder_dominance_lca)
    best_models_occlusion_list = get_best_models_list(models_folder_occlusion)

    # dictionaries matching frames to series and matching series to patients
    dataset_series_dict_rca = dataset_rca.series_dict
    dataset_series_dict_lca = dataset_lca.series_dict
    dataset_study_series_dict_rca = dataset_rca.study_series_dict
    dataset_study_series_dict_lca = dataset_lca.study_series_dict

    # create a table with metrics
    df_metrics = pd.DataFrame(columns = ['precision occlusion', 'recall occlusion', 'precision normal', 'recall normal',
                                            'precision right', 'recall right', 'precision left', 'recall left',
                                            'recall macro projections', 'f1 score', 'matthews corrcoef','accuracy'])

    # go through all partitions of the dataset
    for i in range(5):
        print()
        print(f'***** Fold {i} *****')
        print()

        # create a list of patients to be tested
        test_studies = np.copy(holdout)
        
        test_dataset_rca = Subset(dataset_rca, get_studyid_nums(test_studies, dataset_rca.studyid_ordernum_dict))
        test_dataset_lca = Subset(dataset_lca, get_studyid_nums(test_studies, dataset_lca.studyid_ordernum_dict))

        dataloader_rca = DataLoader(test_dataset_rca, **dataloaders_params['test_dataloader_params'])
        dataloader_lca = DataLoader(test_dataset_lca, **dataloaders_params['test_dataloader_params'])

        print("Occlusion prediction process is working...")
        
        # obtain average patient predictions for occlusion classification models trained on different partitions of the dataset
        study_weighted_preds_dict_occlusion, study_weighted_true_dict_occlusion = get_study_preds_for_model(models_folder_occlusion,
                                                                                        model_occlusion, 
                                                                                        dataloader_rca,
                                                                                        best_models_occlusion_list[i],
                                                                                        dataset_study_series_dict_rca,
                                                                                        model_type = "occlusion",
                                                                                        dataset_series_dict = dataset_series_dict_rca,
                                                                                        device=device)

        # lists of patients for whom occlusion was and was not predicted, respectively
        predicted_occlusion_studies_list = []
        not_predicted_occlusion_studies_list = []

        for study in study_weighted_preds_dict_occlusion.keys():
            if study_weighted_preds_dict_occlusion[study][0] == 1:
                predicted_occlusion_studies_list.append(study)
            else:
                not_predicted_occlusion_studies_list.append(study)

        # list with series for which dominance type predictions should be made on right arteries (no occlusion predicted)
        series_to_predict_by_rca = []
        for study in not_predicted_occlusion_studies_list:
            series_to_predict_by_rca.extend(dataset_study_series_dict_rca[study])

        # list of patients who were not included in the dataset with the right artery type due to lack of good frames defined by the frame quality classification model
        filtered_rca_studies_list = []
        for study in dataset_study_series_dict_lca.keys():
            if study not in dataset_study_series_dict_rca.keys():
                filtered_rca_studies_list.append(study)

        # list of patients whose dominance will be predicted by left-sided arteries
        study_to_predict_by_lca = predicted_occlusion_studies_list + filtered_rca_studies_list

        # list with series for which dominance type predictions should be made by left arteries (occlusion predicted)
        series_to_predict_by_lca = []
        for study in study_to_predict_by_lca:
            if study in dataset_study_series_dict_lca.keys():
                series_to_predict_by_lca.extend(dataset_study_series_dict_lca[study])

        # counting statistics
        print("Number of studies not predicted by occlusion model", len(not_predicted_occlusion_studies_list))
        print("Number of studies predicted by occlusion model", len(predicted_occlusion_studies_list))
        print("Number of studies filtered by RCA model", len(filtered_rca_studies_list))

        print("Dominance RCA prediction process is working...")

        # obtaining patient-averaged predictions for models for right artery type dominance classification trained on different partitions of the dataset
        study_weighted_preds_dict_rca, study_weighted_true_dict_rca = get_study_preds_for_model(models_folder_dominance_rca,
                                                                                        model_dominance, 
                                                                                        dataloader_rca,
                                                                                        best_models_dominance_rca_list[i],
                                                                                        dataset_study_series_dict_rca,
                                                                                        series_to_predict_by_rca,
                                                                                        model_type = "dominance",
                                                                                        dataset_series_dict = dataset_series_dict_rca,
                                                                                        device=device)

        # if occlusion is predicted for the patient, but all frames for the left artery type will be filtered out, then we assign it the right dominance
        for study in predicted_occlusion_studies_list:
            if study in study_weighted_preds_dict_rca.keys():
                study_weighted_preds_dict_rca[study] = [study_weighted_preds_dict_rca[study][0], np.nan, np.nan]

        print("Dominance LCA prediction process is working...")

        # Obtaining patient-averaged predictions for models for left artery type dominance classification trained on different partitions of the dataset
        study_weighted_preds_dict_lca, study_weighted_true_dict_lca = get_study_preds_for_model(models_folder_dominance_lca,
                                                                                        model_dominance, 
                                                                                        dataloader_lca,
                                                                                        best_models_dominance_lca_list[i],
                                                                                        dataset_study_series_dict_lca,
                                                                                        series_to_predict_by_lca,
                                                                                        model_type = "dominance",
                                                                                        dataset_series_dict = dataset_series_dict_lca,
                                                                                        device=device)

        print(len(study_weighted_preds_dict_lca), len(study_to_predict_by_lca))

        # for patients where occlusion was predicted, predict dominance by left-type arteries or by right-type arteries
        for study in predicted_occlusion_studies_list:
            if study not in study_weighted_preds_dict_lca.keys():
                print(f"{study} predicted occlusion and filtered by LCA => assuming right dominance")
                study_weighted_preds_dict_rca[study] = [1, np.nan, np.nan, np.nan]
            else:
                study_weighted_preds_dict_rca[study] = study_weighted_preds_dict_lca[study]

        # for patients where there are no good frames for right-type arteries, predict dominance by left-type arteries if they are present
        for study in filtered_rca_studies_list:
            if study not in study_weighted_preds_dict_lca.keys():
                print(f"{study} was filtered by LCA and RCA")
            else:
                study_weighted_preds_dict_rca[study] = study_weighted_preds_dict_lca[study]
                study_weighted_preds_dict_occlusion[study] = [np.nan, np.nan, np.nan, np.nan]
                study_weighted_true_dict_rca[study] = study_weighted_true_dict_lca[study]
                study_weighted_true_dict_occlusion[study] = [np.nan, np.nan, np.nan]

        # sort all received data by patient folder names
        study_weighted_pred_dict_occlusion = dict(sorted(study_weighted_preds_dict_occlusion.items()))
        study_weighted_true_dict_occlusion = dict(sorted(study_weighted_true_dict_occlusion.items()))
        study_weighted_pred_dict_dominance = dict(sorted(study_weighted_preds_dict_rca.items()))
        study_weighted_true_dict_dominance = dict(sorted(study_weighted_true_dict_rca.items()))

        # get the predictions in a convenient form for recording in a table
        (labels_list_occlusion, 
        confidence_occlusion, 
        std_list_occlusion, 
        entropy_list_occlusion,
        true_list_occlusion,
        true_confidence_occlusion) = get_pred_lists_for_table(study_weighted_pred_dict_occlusion, study_weighted_true_dict_occlusion)

        (labels_list_dominance, 
        confidence_dominance, 
        std_list_dominance, 
        entropy_list_dominance,
        true_list_dominance, 
        true_confidence_dominance) = get_pred_lists_for_table(study_weighted_pred_dict_dominance, study_weighted_true_dict_dominance)

        # create a prediction table
        df_preds = pd.DataFrame({"study": study_weighted_pred_dict_occlusion.keys(),
                                "true_label_occlusion": true_list_occlusion,
                                "preds_occlusion": labels_list_occlusion,
                                "preds_conf_occlusion": confidence_occlusion,
                                "pred_std_occlusion": std_list_occlusion,
                                "preds_entropy_occlusion" : entropy_list_occlusion,
                                "true_label_dominance": true_list_dominance,
                                "preds_dominance": labels_list_dominance,
                                "preds_conf_dominance": confidence_dominance,
                                "pred_std_dominance": std_list_dominance,
                                "preds_entropy_dominance": entropy_list_dominance,})

        # create a folder for logs and save the table to it
        log_folder_path = os.path.join(log_path, f'exp_{exp_number}')

        if not os.path.exists(log_folder_path):
            os.mkdir(log_folder_path)

        log_excel_path = os.path.join(log_folder_path,
                                    f'full_pipeline_study_preds_on_{"_".join(features)}_RCA_LCA_fold_{i}.xlsx')
        df_preds.to_excel(log_excel_path)

        # getting predictions from the table
        y_true_occlusion = np.array(df_preds["true_label_occlusion"].tolist(), dtype=np.float16)
        y_pred_occlusion = np.array(df_preds["preds_occlusion"].tolist(), dtype=np.float16)
        y_true_dominance = np.array(df_preds["true_label_dominance"].tolist(), dtype=np.float16)
        y_pred_dominance = np.array(df_preds["preds_dominance"].tolist(), dtype=np.float16)

        # obtaining separate predictions for different classification tasks to calculate metrics
        y_true_o = np.zeros_like(y_true_occlusion, dtype=np.int8)
        y_true_o[y_true_occlusion == 1] = 1
        y_pred_o = np.zeros_like(y_pred_occlusion, dtype=np.int8)
        y_pred_o[y_pred_occlusion == 1] = 1

        y_true_not_o = np.zeros_like(y_true_occlusion, dtype=np.int8)
        y_true_not_o[y_true_occlusion == 0] = 1
        y_pred_not_o = np.zeros_like(y_pred_occlusion, dtype=np.int8)
        y_pred_not_o[y_pred_occlusion == 0] = 1                                    

        y_true_r = np.zeros_like(y_true_dominance, dtype=np.int8)
        y_true_r[y_true_dominance == 1] = 1
        y_pred_r = np.zeros_like(y_pred_dominance, dtype=np.int8)
        y_pred_r[y_pred_dominance == 1] = 1

        y_true_l = np.zeros_like(y_true_dominance, dtype=np.int8)
        y_true_l[y_true_dominance == 0] = 1
        y_pred_l = np.zeros_like(y_pred_dominance, dtype=np.int8)
        y_pred_l[y_pred_dominance == 0] = 1

        # output intermediate statistics and save metrics to a table
        print(f'precision occlusion {precision_score(y_true_o, y_pred_o)}')
        print(f'recall occlusion {recall_score(y_true_o, y_pred_o)}')
        print(f'precision normal {precision_score(y_true_not_o, y_pred_not_o)}')
        print(f'recall normal {recall_score(y_true_not_o, y_pred_not_o)}')
        print(f'precision right {precision_score(y_true_r, y_pred_r)}')
        print(f'recall right {recall_score(y_true_r, y_pred_r)}')
        print(f'precision left {precision_score(y_true_l, y_pred_l)}')
        print(f'recall left {recall_score(y_true_l, y_pred_l)}')
        print(f'recall macro projections {recall_score(y_true_dominance, y_pred_dominance, average="macro")}')
        print(f'f1 score {f1_score(y_true_dominance, y_pred_dominance, average="macro")}')
        print(f'matthews corrcoef {matthews_corrcoef(y_true_dominance, y_pred_dominance)}')
        print(f'accuracy {accuracy_score(y_true_dominance, y_pred_dominance)}')
        print()

        df_metrics.loc[i, 'precision occlusion'] = precision_score(y_true_o, y_pred_o)
        df_metrics.loc[i, 'recall occlusion'] = recall_score(y_true_o, y_pred_o)
        df_metrics.loc[i, 'precision normal'] = precision_score(y_true_not_o, y_pred_not_o)
        df_metrics.loc[i, 'recall normal'] = recall_score(y_true_not_o, y_pred_not_o)
        df_metrics.loc[i, 'precision right'] = precision_score(y_true_r, y_pred_r)
        df_metrics.loc[i, 'recall right'] = recall_score(y_true_r, y_pred_r)
        df_metrics.loc[i, 'precision left'] = precision_score(y_true_l, y_pred_l)
        df_metrics.loc[i, 'recall left'] = recall_score(y_true_l, y_pred_l)
        df_metrics.loc[i, 'recall macro projections'] = recall_score(y_true_dominance, y_pred_dominance, average="macro")
        df_metrics.loc[i, 'f1 score'] = f1_score(y_true_dominance, y_pred_dominance, average="macro")
        df_metrics.loc[i, 'matthews corrcoef'] = matthews_corrcoef(y_true_dominance, y_pred_dominance)
        df_metrics.loc[i, 'accuracy'] = accuracy_score(y_true_dominance, y_pred_dominance)


    df_metrics.loc[len(df_metrics)] = [None] * df_metrics.shape[1]
    df_metrics.loc[len(df_metrics)] = list(np.mean(df_metrics.loc[:5, df_metrics.columns[:]], axis=0))
    df_metrics.loc[len(df_metrics)] = [None] * df_metrics.shape[1]
    df_metrics.loc[len(df_metrics)] = list(np.std(df_metrics.loc[:5, df_metrics.columns[:]], axis=0))

    log_excel_path = os.path.join(log_folder_path, 
                                f'full_pipeline_test_logs_{"_".join(features)}_RCA_LCA.xlsx')
    df_metrics.to_excel(log_excel_path)


if __name__ == "__main__":
    main()


