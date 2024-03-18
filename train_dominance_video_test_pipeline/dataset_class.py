import numpy as np 
import os
from torch.utils.data import Dataset
import torchvision
import torch
from tqdm import tqdm
import copy
import torch
from torch import Tensor, nn, optim
from torchmetrics.functional import accuracy, f1_score
import torchvision.models.video as tvmv
import sklearn.metrics as skm
import lightning.pytorch as pl
import gc 
from typing import List, Union, Tuple, Optional, Any

import sys
sys.path.insert(1, './')
print('Updated sys.path:', sys.path)

from utils.slices_selection import get_good_idx, predict_good_slices, predict_good_slices_ssim

class HeartLR_simple_main(Dataset):
    def __init__(self, input_path: str, features: List[str] = [], device: str = 'cpu',
                 transform: Optional[str] = None, slice_selection_params: Optional[dict] = None,
                 returns_type: Optional[str] = None, artery_type: Optional[str] = None, 
                 steps: List[int] = [1, 1], seed: int = 42, max_len: Optional[int] = None,
                 proportion_cls: List[int] = [1, 1], status: str ="train", length: int = 32,
                 labels_type: str ="occlusion", n_first_frames_if_empty: bool = False,
                 ssim_if_empty: bool = False, holdout_studies: Optional[List[str]] = None) -> None:
        '''
        Class for the main dataset for the occlusion classification task with new occlusions and left dominantities included in it using video

        input_path - path to the dataset folder
        features - considered features
        device - device on which calculations are performed
        transform - transformations
        slice_selection_params - dictionary with instructions for selecting significant frames
        returns_type - define, return data to solve classification or piplane checking task 
        artery_type - artery type under consideration
        steps - list of steps with which we select frames for left and right dominance
        seed - fixing of random transformations
        max_len - maximum dataset size
        proportion_cls - repetitions of each class
        label_name - defines the type of feature we are classifying, either occlusions or artifacts
        status - determines whether to create a dataset for training or test phase
        length - length of the training video
        n_first_frames_if_empty - flag determines whether we should apply the n_first_frames method if all frames are filtered by the model
        ssim_if_empty - flag that determines whether to apply ssim method if all frames are filtered by the model.
        holdout_studies - list of patients put in hold out to test models on them
        
        return: None
        '''

        super().__init__()

        # dictionary, where key is an ordinal number, values are series
        self.series_dict = {}
        
        # unique series number
        self.series_uniq_num = 0
        
        # dictionary, where key is the patient number, values are a list of series for the patient
        self.study_series_dict ={}
        
        # set of the patients
        self.studies_set = set()
        
        # main list for the dataset with information for each video
        self.projection_list = []
        
        self.steps = steps
        self.status = status
        self.proportion_cls = proportion_cls
        
        # unification of the list of arteries considered in the study
        if artery_type == "LCA":
            self.artery_type = ["LCA"]
        elif artery_type == "RCA":
            self.artery_type = ["RCA"]
        elif artery_type == "LCA/RCA":
            self.artery_type = ["LCA", "RCA"]
        else:
            raise TypeError(f"Wrong artery type, {artery_type}")

        self.slice_selection_params = slice_selection_params
        self.device = device
        self.max_len = max_len

        # dictionary, where key is the patient number, values are the list of frames belonging to it
        self.studyid_ordernum_dict = {}
        self.features = features

        # counters for statistics
        self.all_pictures = 0
        self.good_pictures = 0
        self.bad_pictures = 0
        self.count_label_0 = 0
        self.count_label_1 = 0

        self.n_first_frames_if_empty = n_first_frames_if_empty
        self.ssim_if_empty = ssim_if_empty
        self.holdout_studies = holdout_studies

        self.length = length
        self.labels_type = labels_type
        self.returns_type = returns_type

        # if we need to use the model for slice selection, load it
        if self.slice_selection_params['frames_selection_method'] == "model":
            self.slice_selection_model = torchvision.models.convnext_tiny(weights=None, num_classes=2)
            self.slice_selection_model.load_state_dict(torch.load(slice_selection_params['model_slice_selection_path'], map_location=device))
            self.slice_selection_model.to(device)
            self.slice_selection_model.train(False)

        # dataset loading and processing
        self.load_data(input_path)
        self.transform = transform

        # output intermediate statistics
        print(f"Raw pictures: {self.all_pictures}, good: {self.good_pictures}, bad: {self.bad_pictures}, proportion: {self.good_pictures/self.all_pictures}")
        print("Сколько Left_Dominance:", self.count_label_0)
        print("Сколько Right_Dominance:", self.count_label_1)

        self.seed = seed

        # deleting the model from memory and clearing the cache
        del self.slice_selection_model
        gc.collect()
        torch.cuda.empty_cache() 

    def load_data(self, input_path: str) -> None:
        '''
        Loading and processing data for the dataset

        input_path - path to the dataset folder

        return: None
        '''

        print("Dataset is creating...")

        # go over all the features in consideration
        for feature in self.features:

            # stop loading if the maximum dataset size is exceeded
            if self.max_len is not None and len(self.projection_list) >= self.max_len:
                break
            print(f"Current directory is {feature}")

            # the path to the folder with the left heart type
            if os.path.exists(os.path.join(input_path, feature, "Left_Dominance")):
                left_type = os.path.join(input_path, feature, "Left_Dominance")  
                print("Left_Dominance")

                # loading data for left dominance
                self.read_one_type(left_type)

            # the path to the folder with the right heart type
            if os.path.exists(os.path.join(input_path, feature, "Right_Dominance")):
                right_type = os.path.join(input_path, feature, "Right_Dominance") 
                print("Right_Dominance")   

                # loading data for right dominance
                self.read_one_type(right_type)
    
    def read_one_type(self, path_to_type: str) -> None:
        '''
        Function to read one type of hearts

        path_to_type - path to the folder with a certain type 

        return: None
        '''

        print("Method:", self.slice_selection_params['frames_selection_method'])

        # dominance type in abbreviated form
        heart_type = os.path.basename(path_to_type).lower()[:4]

        # patient folders
        study_folders = os.listdir(path_to_type)

        for study in tqdm(study_folders):

            # when testing for hold out, ignore patients who are not included in it
            if self.holdout_studies is not None and study not in self.holdout_studies:
                continue

            study_path = os.path.join(path_to_type, study)
            
            for artery_type in os.listdir(study_path):

                # ignore the types of arteries that need to be skipped
                if artery_type not in self.artery_type:
                    continue
                
                # path to the folder with projections of a certain artery type
                artery_path = os.path.join(study_path, artery_type)

                files = os.listdir(artery_path)

                # go through all the npz
                for file in files:
                    
                    # receiving information for npz
                    file_path = os.path.join(artery_path, file)
                    series = os.path.basename(file_path).replace(".npz", "")
                    hearth_npz_info = np.load(file_path)
                    heart_projection = hearth_npz_info["pixel_array"]
                    # initial_dcm_path = hearth_npz_info["pixel_path"]
                    study_num = study.split("_")[-1]

                    if study_num not in self.studies_set:
                        self.studies_set.add(study_num)
                    
                    # filling dictionaries
                    if series not in self.series_dict.values():
                        self.series_dict[self.series_uniq_num] = series
                        cur_series_num = self.series_uniq_num
                        self.series_uniq_num += 1
                    else:
                        cur_series_num = list(self.series_dict.keys())[list(self.series_dict.values()).index(series)]

                    # get feature information
                    is_collaterals = hearth_npz_info["is_collaterals"]
                    is_occlusion = hearth_npz_info["is_occlusion"]
                    is_undefined_type = hearth_npz_info["is_undefined_type"]

                    # conditions in accordance with medicine
                    if artery_type == "RCA":
                        is_collaterals = False
                    elif artery_type == "LCA":
                        is_occlusion = False
                    else:
                        raise ValueError(f"Wrong artery type, {artery_type}")
                    
                    # checking that we are using the video and not the frame
                    if len(heart_projection.shape) == 2:
                        heart_projection = np.expand_dims(heart_projection, 0)

                    # counting statistics
                    self.all_pictures += len(heart_projection)
                    self.bad_pictures += len(heart_projection)

                    if self.slice_selection_params['frames_selection_method'] == "model":

                        # selection of slices using the model
                        if self.status == "train":
                            if self.labels_type == "dominance":
                                if heart_type == 'left':
                                    n_repeat = self.proportion_cls[0]
                                    step = self.steps[0]
                                    conf_thr = self.slice_selection_params['conf_thr_slices_left']
                                elif heart_type == 'righ':
                                    n_repeat = self.proportion_cls[0]
                                    step = self.steps[1]
                                    conf_thr = self.slice_selection_params['conf_thr_slices_right']
                            
                            elif self.labels_type == "occlusion":
                                if not is_occlusion:
                                    n_repeat = self.proportion_cls[0]
                                    step = self.steps[0]
                                    conf_thr = self.slice_selection_params['conf_thr_slices_not_occlusion']
                                elif is_occlusion:
                                    n_repeat = self.proportion_cls[0]
                                    step = self.steps[1]
                                    conf_thr = self.slice_selection_params['conf_thr_slices_occlusion']

                        elif self.status == "test":
                            conf_thr = self.slice_selection_params['conf_thr_slices'] 
                        else:
                            raise AssertionError(f"Wrong status: {self.status}")
                        
                        preds = predict_good_slices(heart_projection, model=self.slice_selection_model, device=self.device)
                        good_i = get_good_idx(preds, conf_thr, **self.slice_selection_params)

                        # if necessary, apply the n_first_frames method if the model has filtered all frames
                        if good_i == [] and self.n_first_frames_if_empty:
                            print(len(heart_projection), study)
                            good_i = list(range(len(heart_projection)))
                            good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']:]
                            heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']:]

                        # if necessary apply ssim method if the model has filtered all frames 
                        elif good_i == [] and self.ssim_if_empty:
                            print(len(heart_projection), study)
                            good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                            heart_projection = heart_projection[good_i]
                        else:
                            heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "ssim":
                        # selection of slices using structural similarity
                        good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                        heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "first_frames" and \
                        self.slice_selection_params['n_first_frames_to_skip'] > 0:
                        # selection of slices by discarding the first n slices
                        good_i = list(range(len(heart_projection)))[self.slice_selection_params['n_first_frames_to_skip']: ]
                        heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']: ]
                   
                    
                    elif self.slice_selection_params['frames_selection_method'] is None:
                        good_i = list(range(len(heart_projection)))

                    else:
                        raise AssertionError(f"Wrong frames selection method: {self.slice_selection_params['frames_selection_method']}")

                    # counting statistics
                    self.good_pictures += len(good_i)
                    self.bad_pictures -= len(good_i)

                    # checking that we are using the video and not the frame
                    assert len(heart_projection.shape) == 3

                    if len(heart_projection) == 0:
                        continue
                        
                    # adding information for each video to the dataset
                    project_dict = {"dominance_type": heart_type, "artery_type": artery_type, 'series_num': cur_series_num,\
                                    "video": heart_projection, \
                                    "is_collaterals": is_collaterals, "is_occlusion": is_occlusion, \
                                    "is_undefined_type": is_undefined_type, 'studyid': study_num, 'seriesid': series}
                    
                    # filling in relevant dictionaries and counting statistics
                    if study not in self.study_series_dict.keys():
                        self.study_series_dict[study] = [series]
                    else:
                        self.study_series_dict[study].append(series)

                    if study not in self.studyid_ordernum_dict.keys():
                        self.studyid_ordernum_dict[study] = [len(self.projection_list)]
                    else:
                        self.studyid_ordernum_dict[study].append(len(self.projection_list))
                    self.projection_list.append(project_dict)
                    
                    if self.labels_type == "dominance":
                        if project_dict['dominance_type'] == 'left':
                            self.count_label_0 += 1
                        elif project_dict['dominance_type'] == 'righ':
                            self.count_label_1 += 1
                        else:
                            raise AssertionError(f"Wrong dominance type: {project_dict['dominance_type']}")
                        
                    elif self.labels_type == "occlusion":
                            if project_dict['is_occlusion'] == True:
                                self.count_label_1 += 1
                            elif project_dict['is_occlusion'] == False:
                                self.count_label_0 += 1
                            else:
                                raise AssertionError(f"Wrong occlusion type: {project_dict['is_occlusion']}")

                    # stop when the dataset size is exceeded
                    if self.max_len is not None and len(self.projection_list) >= self.max_len:
                        return


    def __len__(self) -> int:
        return len(self.projection_list)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:

        # receiving a video with the appropriate transformations and labels
        proj_i = self.projection_list[idx]  
        
        video = proj_i['video']

        # repeat short videos
        while len(video) < self.length:
            video = np.concatenate([video, video])

        # when practicing leave the set number of frames
        t = len(video)
        if self.status == "train":
            begin = torch.randint(low=0, high=t-self.length+1, size=(1,))
            end = begin + self.length
            video = video[begin:end, :, :]
        else:
            video = video 
            
        video = np.array(video, dtype=np.uint8)
        video = torch.tensor(np.stack([video, video, video], axis=-1))

        if self.transform is not None:
            video = self.transform(video) 

        # if we test the pipelines, we return labels for occlusion and dominance
        if self.returns_type == "pipeline":
            if proj_i['is_occlusion'] == False:
                label_occlusion = torch.tensor(0).type(torch.float)
            elif proj_i['is_occlusion'] == True:
                label_occlusion = torch.tensor(1).type(torch.float)
            
            if proj_i['dominance_type'] == 'left':
                label_dominance = torch.tensor(0).type(torch.float)
            elif proj_i['dominance_type'] == 'righ':
                label_dominance = torch.tensor(1).type(torch.float)
            return video, label_occlusion, label_dominance, proj_i['series_num']

        # return labels depending on the current occlusion or dominance classification task
        else:
            if self.labels_type == "dominance":
                if proj_i['dominance_type'] == 'left':
                    label = torch.tensor([0]).type(torch.float)
                elif proj_i['dominance_type'] == 'righ':
                    label = torch.tensor([1]).type(torch.float)
            elif self.labels_type == "occlusion":
                if proj_i['is_occlusion'] == False:
                    label = torch.tensor([0]).type(torch.float)
                elif proj_i['is_occlusion'] == True:
                    label = torch.tensor([1]).type(torch.float)
            
            return video, label
        


class HeartLR_simple_test_labeled(Dataset):
    def __init__(self, input_path: str, device: str = 'cpu', transform: Optional[str] = None,
                 slice_selection_params: Optional[dict] = None, returns_type: Optional[str] = None,
                 artery_type: Optional[str] = None, steps: List[int] = [1, 1], seed: int = 42,
                 max_len: Optional[int] = None, proportion_cls: List[int] = [1, 1],
                 n_first_frames_if_empty: bool = False,
                 ssim_if_empty: bool = False, length: int = 32, status: str = "test") -> None:
        '''
        Class for dataset for testing models on domain shift dataset with the assumption that there are no patients with occlusions inside it with data in video form

        input_path - path to the folder with the dataset
        device - device on which calculations are performed
        transform - transforms
        slice_selection_params - dictionary with instructions for selecting significant frames
        returns_type - define whether to return data for classification task or to check a pipeline 
        artery_type - artery types to be considered
        steps - list of steps with which we select frames for left and right dominance
        seed - fixing of random transformations
        max_len - maximum dataset size
        proportion_cls - repetitions of each class
        n_first_frames_if_empty - flag, determines whether to apply n_first_frames method if all frames are filtered by the model
        ssim_if_empty - flag that determines whether to apply ssim method if all frames are filtered by the model.
        length - length of video during training
        status - determines whether it is necessary to create a dataset for training or test phase
        
        return: None
        '''

        super().__init__()

        # dictionary, where key is the patient number, values are a list of series for the patient
        self.study_series_dict = {}

        # dictionary, where key is an ordinal number, values are series
        self.series_dict = {}
        
        # unique series number
        self.series_uniq_num = 0

        self.returns_type = returns_type

       # main list for the dataset with information for each video
        self.projection_list = []
        self.steps = steps

        self.proportion_cls = proportion_cls

        # unification of the list of arteries considered in the study
        if artery_type == "LCA":
            self.artery_type = ["LCA"]
        elif artery_type == "RCA":
            self.artery_type = ["RCA"]
        elif artery_type == "LCA/RCA":
            self.artery_type = ["LCA", "RCA"]
        else:
            raise TypeError(f"Wrong artery type, {artery_type}")
        
        self.slice_selection_params = copy.deepcopy(slice_selection_params)
        self.device = device
        self.max_len = max_len

        # dictionary, where key is the patient number, values are the list of frames belonging to it
        self.studyid_ordernum_dict = {}

        # counters for statistics
        self.all_pictures = 0
        self.good_pictures = 0
        self.bad_pictures = 0

        self.n_first_frames_if_empty = n_first_frames_if_empty
        self.ssim_if_empty = ssim_if_empty

        self.length = length
        self.status = status

        # if we need to use the model for slice selection, load it
        if self.slice_selection_params['frames_selection_method'] == "model":
            self.slice_selection_model = torchvision.models.convnext_tiny(weights=None, num_classes=2)
            self.slice_selection_model.load_state_dict(torch.load(slice_selection_params['model_slice_selection_path'], map_location=device))
            self.slice_selection_model.to(device)
            self.slice_selection_model.train(False)

        # dataset loading and processing
        self.load_data(input_path)
        self.transform = transform

        # output intermediate statistics
        print(f"Raw pictures: {self.all_pictures}, good: {self.good_pictures}, bad:{self.bad_pictures}, proportion:{self.good_pictures/self.all_pictures}")

        self.seed = seed

        # Deleting the model and clearing the cache
        del self.slice_selection_model
        gc.collect()
        torch.cuda.empty_cache() 

    def load_data(self, input_path: str) -> None:
        '''
        Loading and processing data for the dataset

        input_path - path to the dataset folder

        return: None
        '''

        # the path to the folder with the left heart type
        if os.path.exists(os.path.join(input_path, "Left_Dominance")):
            left_type = os.path.join(input_path, "Left_Dominance")  
            print("Left_Dominance")

            # loading data for left dominance
            self.read_one_type(left_type)

        # path to the folder with the right heart type
        if os.path.exists(os.path.join(input_path, "Right_Dominance")):
            right_type = os.path.join(input_path, "Right_Dominance") 
            print("Right_Dominance")   

            # loading data for right dominance
            self.read_one_type(right_type)     
    
    def read_one_type(self, path_to_type: str) -> None:
        '''
        Function to read one type of hearts

        path_to_type - path to the folder with a certain type 

        return: None
        '''

        # dominance type in abbreviated form
        heart_type = os.path.basename(path_to_type).lower()[:4]

        # patient folders
        study_folders = os.listdir(path_to_type)

        for study in tqdm(study_folders):
            study_path = os.path.join(path_to_type, study)

            for artery_type in os.listdir(study_path):

                # ignore the types of arteries that need to be skipped
                if artery_type not in self.artery_type:
                    continue
                
                # path to the folder with projections of a certain artery type 
                artery_path = os.path.join(study_path, artery_type)
                
                files = os.listdir(artery_path)
                
                # go through all the npz
                for file in files:  
                    
                    # receiving information for npz
                    file_path = os.path.join(artery_path, file)
                    series = os.path.basename(file_path).replace(".npz", "")
                    hearth_npz_info = np.load(file_path)
                    heart_projection = hearth_npz_info["pixel_array"]
                    # initial_dcm_path = hearth_npz_info["pixel_path"]

                    # filling dictionaries
                    if series not in self.series_dict.values():
                        self.series_dict[self.series_uniq_num] = series
                        cur_series_num = self.series_uniq_num
                        self.series_uniq_num += 1
                    else:
                        cur_series_num = list(self.series_dict.keys())[list(self.series_dict.values()).index(series)]

                    study_num = study.split("_")[-1]
                    
                    # checking that we are using the video and not the frame
                    if len(heart_projection.shape) == 2:
                        heart_projection = np.expand_dims(heart_projection, 0)

                    # counting statistics
                    self.all_pictures += len(heart_projection)
                    self.bad_pictures += len(heart_projection)

                    if self.slice_selection_params['frames_selection_method'] == "model":

                        # selection of slices using the model
                        n_repeat = 1
                        conf_thr = self.slice_selection_params['conf_thr_slices'] 
                        step = 1

                        preds = predict_good_slices(heart_projection, model=self.slice_selection_model, device=self.device)
                        good_i = get_good_idx(preds, conf_thr, **self.slice_selection_params)

                        # if necessary, apply the n_first_frames method if the model has filtered all frames
                        if good_i == [] and self.n_first_frames_if_empty:
                            good_i = list(range(len(heart_projection)))
                            good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']: ]
                            heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']: ]

                        # if necessary apply ssim method if the model has filtered all frames
                        elif good_i == [] and self.ssim_if_empty:
                            print(len(heart_projection), study)
                            good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                            heart_projection = heart_projection[good_i]
                        else:
                            heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "ssim":
                        # selection of slices using structural similarity
                        self.slice_selection_params["window"] = self.slice_selection_params["window"] - 1

                        good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                        heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "first_frames" and \
                        self.slice_selection_params['n_first_frames_to_skip'] > 0:
                        # selection of slices by discarding the first n slices
                        good_i = list(range(len(heart_projection)))
                        good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']: ]
                        heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']: ]

                    elif self.slice_selection_params['frames_selection_method'] is None:
                        good_i = list(range(len(heart_projection)))

                    else:
                        raise AssertionError(f"Wrong frames selection method: {self.slice_selection_params['frames_selection_method']}")

                    # counting statistics
                    self.good_pictures += len(good_i)
                    self.bad_pictures -= len(good_i)

                    # checking that we are using the video and not the frame
                    assert len(heart_projection.shape) == 3

                    if len(heart_projection) == 0:
                        continue
                    
                    # adding information for each video to the dataset
                    project_dict = {"dominance_type": heart_type, "artery_type": artery_type, 'series_num': cur_series_num,\
                                    "video": heart_projection, \
                                    'studyid': study_num, 'seriesid': series}
                    
                    # filling in relevant dictionaries 
                    if study not in self.study_series_dict.keys():
                        self.study_series_dict[study] = [series]
                    else:
                        self.study_series_dict[study].append(series)

                    if study not in self.studyid_ordernum_dict.keys():
                        self.studyid_ordernum_dict[study] = [len(self.projection_list)]
                    else:
                        self.studyid_ordernum_dict[study].append(len(self.projection_list))
                    self.projection_list.append(project_dict)

                    # stop when the dataset size is exceeded
                    if self.max_len is not None and len(self.projection_list) >= self.max_len:
                        return

    def __len__(self) -> int:
        return len(self.projection_list)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
                                                           Tuple[torch.Tensor, torch.Tensor, int]]:
        
        # receiving a frame with the appropriate transformations and labels
        proj_i = self.projection_list[idx]  
        
        video = proj_i['video']

        # if the video's short, we repeat it 
        while len(video) < self.length:
            video = np.concatenate([video, video])
        t = len(video)

        # when practicing leave the set number of frames
        if self.status == "train":
            begin = torch.randint(low=0, high=t-self.length+1, size=(1,))
            end = begin + self.length
            video = video[begin:end, :, :]
        else:
            video = video 

        video = np.array(video, dtype=np.uint8)
        video = torch.tensor(np.stack([video, video, video], axis=-1))

        if self.transform is not None:
            video = self.transform(video) 

        # if we test the pipelines, we return labels for occlusion and dominance
        if self.returns_type == "pipeline":
            label_occlusion = torch.tensor(0).type(torch.float)
            
            if proj_i['dominance_type'] == 'left':
                label_dominance = torch.tensor(0).type(torch.float)
            elif proj_i['dominance_type'] == 'righ':
                label_dominance = torch.tensor(1).type(torch.float)
            return video, label_occlusion, label_dominance, proj_i['series_num']

        # return labels depending on the current occlusion or dominance classification task
        else:
            if self.label_name == "occlusion":
                if proj_i['is_occlusion'] == False:
                    label = torch.tensor(0).type(torch.float)
                elif proj_i['is_occlusion'] == True:
                    label = torch.tensor(1).type(torch.float)
            elif self.label_name == "artifact":
                if proj_i['is_artifact'] == False:
                    label = torch.tensor(0).type(torch.float)
                elif proj_i['is_artifact'] == True:
                    label = torch.tensor(1).type(torch.float)
            else:
                raise AssertionError(f"Wrong label:{self.label_name}")

        return video, label, proj_i['series_num']


class HeartLR_simple_main_frames(Dataset):
    def __init__(self, input_path: str, features: List[str] = [], device: str = 'cpu',
                  transform: Optional[str] = None, slice_selection_params: Optional[dict] = None,
                 returns_type: Optional[str] = None, artery_type: Optional[str] = None, 
                 steps: List[int] = [1, 1], seed: int = 42, max_len: Optional[int] = None, 
                 proportion_cls: List[int] = [1, 1], label_name: str = "occlusion", 
                 status: str = "train", n_first_frames_if_empty: bool = False,
                 ssim_if_empty: bool = False, holdout_studies: Optional[List[str]] = None) -> None:
        '''
        Class for the main dataset for the occlusion and dominance classification task with new occlusions and left dominances included there with data in the form of frames

        input_path - path to the folder with the dataset
        features - considered features
        device - device on which calculations are performed
        transform - transformations
        slice_selection_params - dictionary with instructions for selecting significant frames
        returns_type - define, to return data for classification or piplane checking task 
        artery_type - artery type under consideration
        steps - list of steps with which we select frames for left and right dominance
        seed - fixing of random transformations
        max_len - maximum dataset size
        proportion_cls - repetitions of each class
        label_name - defines the type of feature we are classifying, either occlusions or artifacts
        status - determines whether to create a dataset for training or test phase
        n_first_frames_if_empty - flag determines whether we should apply the n_first_frames method if all frames are filtered by the model
        ssim_if_empty - flag that determines whether to apply ssim method if all frames are filtered by the model.
        holdout_studies - list of patients put in holdout to test models on them
        
        return: None
        '''

        super().__init__()

        # dictionary, where key is the patient number, values are a list of series for the patient
        self.study_series_dict = {}

        # dictionary, where key is an ordinal number, values are series
        self.series_dict = {}

        # unique series number
        self.series_uniq_num = 0

        self.status = status

        self.label_name=label_name
        assert self.label_name in ["occlusion", "artifact"], f"Undefined label type:{self.label_name}"
        print(f"y_labels correspond to the {self.label_name}")

        self.returns_type = returns_type

        # main list for the dataset with information for each frame
        self.projection_list = []

        self.steps = steps
        self.proportion_cls = proportion_cls

        # unification of the list of arteries considered in the study
        if artery_type == "LCA":
            self.artery_type = ["LCA"]
        elif artery_type == "RCA":
            self.artery_type = ["RCA"]
        elif artery_type == "LCA/RCA":
            self.artery_type = ["LCA", "RCA"]
        else:
            raise TypeError(f"Wrong artery type, {artery_type}")
        

        self.slice_selection_params = copy.deepcopy(slice_selection_params)
        self.device = device
        self.max_len = max_len

        # dictionary, where key is the patient number, values are the list of frames belonging to it
        self.studyid_ordernum_dict = {}
        self.features = features

        # counters for statistics
        self.all_pictures = 0
        self.good_pictures = 0
        self.bad_pictures = 0
        self.count_label_0 = 0
        self.count_label_1 = 0

        self.n_first_frames_if_empty = n_first_frames_if_empty
        self.ssim_if_empty = ssim_if_empty

        self.holdout_studies = holdout_studies

        # if we need to use the model for slice selection, load it
        if self.slice_selection_params['frames_selection_method'] == "model":
            self.slice_selection_model = torchvision.models.convnext_tiny(weights=None, num_classes=2)
            self.slice_selection_model.load_state_dict(torch.load(slice_selection_params['model_slice_selection_path'], map_location=device))
            self.slice_selection_model.to(device)
            self.slice_selection_model.train(False)

        # dataset loading and processing
        self.load_data(input_path)
        self.transform = transform

        # output intermediate statistics
        print(f"Raw pictures: {self.all_pictures}, good: {self.good_pictures}, bad:{self.bad_pictures}, proportion:{self.good_pictures/self.all_pictures}")
        print("How many occlusions:", self.count_label_1)
        print("How many normal:", self.count_label_0)

        self.seed = seed

        # deleting the model from memory and clearing the cache
        del self.slice_selection_model
        gc.collect()
        torch.cuda.empty_cache() 

    def load_data(self, input_path: str) -> None:
        '''
        Loading and processing data for the dataset

        input_path - path to the dataset folder
        
        return: None
        '''

        # go over all the features in consideration
        for feature in self.features:
            if self.max_len is not None and len(self.projection_list) >= self.max_len:
                break

            print(f"Current directory is {feature}")

            # the path to the folder with the left heart type
            if os.path.exists(os.path.join(input_path, feature, "Left_Dominance")):
                left_type = os.path.join(input_path, feature, "Left_Dominance")  
                print("Left_Dominance")

                # loading data for left dominance
                self.read_one_type(left_type)

            # the path to the folder with the right heart type
            if os.path.exists(os.path.join(input_path, feature, "Right_Dominance")):
                right_type = os.path.join(input_path, feature, "Right_Dominance") 
                print("Right_Dominance")  

                # loading data for right dominance
                self.read_one_type(right_type)

    
    def read_one_type(self, path_to_type: str) -> None:
        '''
        Function to read one type of hearts

        path_to_type - path to the folder with a certain type 
        
        return: None
        '''

        print("Method:", self.slice_selection_params['frames_selection_method'])

        # dominance type in abbreviated form
        heart_type = os.path.basename(path_to_type).lower()[:4]

        # patient folders
        study_folders = os.listdir(path_to_type)

        for study in tqdm(study_folders):

            # when testing for hold out, ignore patients who are not included in it
            if self.holdout_studies is not None and study not in self.holdout_studies:
                continue

            study_path = os.path.join(path_to_type, study)
            
            for artery_type in os.listdir(study_path):

                # ignore the types of arteries that need to be skipped
                if artery_type not in self.artery_type:
                    continue
                
                # path to the folder with projections of a certain artery type
                artery_path = os.path.join(study_path, artery_type)

                files = os.listdir(artery_path)

                # go through all the npz
                for file in files:
                    
                    # receiving information for npz
                    file_path = os.path.join(artery_path, file)
                    series = os.path.basename(file_path).replace(".npz", "")
                    hearth_npz_info = np.load(file_path)
                    heart_projection = hearth_npz_info["pixel_array"]
                    # initial_dcm_path = hearth_npz_info["pixel_path"]
                    study_num = study.split("_")[-1]

                    # filling dictionaries
                    if series not in self.series_dict.values():
                        self.series_dict[self.series_uniq_num] = series
                        cur_series_num = self.series_uniq_num
                        self.series_uniq_num += 1
                    else:
                        cur_series_num = list(self.series_dict.keys())[list(self.series_dict.values()).index(series)]
                    
                    # get feature information from the table
                    is_collaterals = hearth_npz_info["is_collaterals"]
                    is_occlusion = hearth_npz_info["is_occlusion"]
                    is_undefined_type = hearth_npz_info["is_undefined_type"]
                    is_artifact = hearth_npz_info["is_artifact"]

                    # check the assigned feature types according to the dataset design
                    if path_to_type.split("/")[-2] == 'artifact':
                        assert is_artifact, f"Wrong is_artifact variable:{is_artifact}, current type:{path_to_type.split('/')[-2]}"
                    else:
                        assert not is_artifact, f"Wrong is_artifact variable:{is_artifact}, current type:{path_to_type.split('/')[-2]}, {study}"

                    # conditions in accordance with medicine
                    if artery_type == "RCA":
                        is_collaterals = False
                    elif artery_type == "LCA":
                        is_occlusion = False
                    else:
                        raise ValueError(f"Wrong artery type, {artery_type}")
                    
                    # checking that we are using the video and not the frame
                    if len(heart_projection.shape) == 2:
                        heart_projection = np.expand_dims(heart_projection, 0)

                    # counting statistics
                    self.all_pictures += len(heart_projection)
                    self.bad_pictures += len(heart_projection)

                    if self.slice_selection_params['frames_selection_method'] == "model":
                        # selection of slices using the model

                        if self.status == "train":
                        
                            if not is_occlusion:
                                n_repeat = self.proportion_cls[0]
                                step = self.steps[0]
                                conf_thr = self.slice_selection_params['conf_thr_slices_not_occlusion']
                            elif is_occlusion:
                                n_repeat = self.proportion_cls[0]
                                step = self.steps[1]
                                conf_thr = self.slice_selection_params['conf_thr_slices_occlusion']

                        elif self.status == "test":
                            n_repeat = 1
                            conf_thr = self.slice_selection_params['conf_thr_slices'] 
                            step = 1

                        else:
                            assert ValueError(f"Wrong status: {self.status}")

                        preds = predict_good_slices(heart_projection, model=self.slice_selection_model, device=self.device)
                        good_i = get_good_idx(preds, conf_thr, **self.slice_selection_params)

                        # if necessary, apply the n_first_frames method if the model has filtered all frames
                        if good_i == [] and self.n_first_frames_if_empty:
                            print(len(heart_projection), study)
                            good_i = list(range(len(heart_projection)))
                            good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']:]
                            heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']:]

                        # if necessary apply ssim method if the model has filtered all frames
                        elif good_i == [] and self.ssim_if_empty:
                            print(len(heart_projection), study)
                            good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                            heart_projection = heart_projection[good_i]
                        else:
                            heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "ssim":
                        # selection of slices using structural similarity
                        if is_occlusion:
                            self.slice_selection_params["window"] = self.slice_selection_params["window"] + 1
                        else:
                            self.slice_selection_params["window"] = self.slice_selection_params["window"] - 1
                        # print(is_occlusion, self.slice_selection_params["window"])
                        good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                        heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "first_frames" and \
                        self.slice_selection_params['n_first_frames_to_skip'] > 0:
                        # selection of slices by discarding the first n slices
                        good_i = list(range(len(heart_projection)))
                        good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']: ]
                        heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']: ]

                    elif self.slice_selection_params['frames_selection_method'] is None:
                        good_i = list(range(len(heart_projection)))

                    else:
                        raise AssertionError(f"Wrong frames selection method: {self.slice_selection_params['frames_selection_method']}")
                    
                    # counting statistics
                    self.good_pictures += len(good_i)
                    self.bad_pictures -= len(good_i)

                    # checking that we are using the video and not the frame
                    assert len(heart_projection.shape) == 3

                    n_repeat = 1 

                    # adding information for each frame to the dataset
                    for j in range(n_repeat):
 
                        for i, frame_num in enumerate(good_i):  
              
                            project_dict = {"dominance_type": heart_type, "artery_type": artery_type, "is_artifact": is_artifact,\
                                            "image": heart_projection[i], 'frame_num': frame_num, \
                                            "is_collaterals": is_collaterals, "is_occlusion": is_occlusion, \
                                            "is_undefined_type": is_undefined_type, 'studyid': study_num, 'seriesid': series,
                                            'series_num': cur_series_num}
                            
                            # filling in relevant dictionaries and counting statistics
                            if study not in self.study_series_dict.keys():
                                self.study_series_dict[study] = [series]
                            else:
                                self.study_series_dict[study].append(series)

                            if study not in self.studyid_ordernum_dict.keys():
                                self.studyid_ordernum_dict[study] = [len(self.projection_list)]
                            else:
                                self.studyid_ordernum_dict[study].append(len(self.projection_list))
                            self.projection_list.append(project_dict)

                            if project_dict['dominance_type'] == "righ":
                                self.count_label_1 += 1
                            elif project_dict['dominance_type'] == "left":
                                self.count_label_0 += 1
                            else:
                                raise AssertionError(f"Wrong occlusion type: {project_dict['is_occlusion']}")

                            # stop when the dataset size is exceeded
                            if self.max_len is not None and len(self.projection_list) >= self.max_len:
                                return

    def __len__(self) -> int:
        return len(self.projection_list)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[torch.Tensor, torch.Tensor, int], 
                                                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        
        # receiving a frame with the appropriate transformations and labels
        proj_i = self.projection_list[idx]  
        
        img = proj_i['image']
        img = np.expand_dims(img, -1)
        img = np.concatenate([img, img, img], -1)
        img = np.array(img, dtype=np.uint8)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img / 255

        # if we test the pipelines, we return labels for occlusion and dominance
        if self.returns_type == "pipeline":
            if proj_i['is_occlusion'] == False:
                label_occlusion = torch.tensor(0).type(torch.float)
            elif proj_i['is_occlusion'] == True:
                label_occlusion = torch.tensor(1).type(torch.float)
            
            if proj_i['dominance_type'] == 'left':
                label_dominance = torch.tensor(0).type(torch.float)
            elif proj_i['dominance_type'] == 'righ':
                label_dominance = torch.tensor(1).type(torch.float)
            return img, label_occlusion, label_dominance, proj_i['series_num']

        # return labels depending on the current occlusion or dominance classification task
        else:
            if self.label_name == "occlusion":
                if proj_i['is_occlusion'] == False:
                    label = torch.tensor(0).type(torch.float)
                elif proj_i['is_occlusion'] == True:
                    label = torch.tensor(1).type(torch.float)
            elif self.label_name == "artifact":
                if proj_i['is_artifact'] == False:
                    label = torch.tensor(0).type(torch.float)
                elif proj_i['is_artifact'] == True:
                    label = torch.tensor(1).type(torch.float)
            else:
                raise AssertionError(f"Wrong label:{self.label_name}")

        return img, label, proj_i['series_num'] 


class HeartLR_simple_test_labeled_frames(Dataset):
    def __init__(self, input_path: str, device: str = 'cpu', transform: Optional[str] = None,
                 slice_selection_params: Optional[dict] = None, returns_type: Optional[str] = None,
                 artery_type: Optional[str] = None, steps: List[int] = [1, 1], seed: int = 42, 
                 max_len: Optional[int] = None, proportion_cls: List[int] = [1, 1],
                 n_first_frames_if_empty: bool = False, ssim_if_empty: bool = False) -> None:
        '''
        Class for dataset for testing models on domain shift dataset with the assumption that there are no patients with occlusions inside it

        input_path - path to the dataset folder
        device - device on which calculations are performed
        transform - transformations
        slice_selection_params - dictionary with instructions for selecting significant frames
        returns_type - define whether to return data for classification task or to check a pipeline 
        artery_type - artery types to be considered
        steps - list of steps with which we select frames for left and right dominance
        seed - fixing of random transformations
        max_len - maximum dataset size
        proportion_cls - repetitions of each class
        new_occlusion_path - path to dataset with new occlusions
        n_first_frames_if_empty - flag, determines whether to apply n_first_frames method, if all frames are filtered by the model
        ssim_if_empty - flag that determines whether to apply ssim method if all frames are filtered by the model.
        
        return: None
        '''

        super().__init__()

        # dictionary, where key is the patient number, values are a list of series for the patient
        self.study_series_dict = {}

        # dictionary, where key is an ordinal number, values are series
        self.series_dict = {}

        # unique series number
        self.series_uniq_num = 0

        self.returns_type = returns_type

        # main list for the dataset with information for each frame
        self.projection_list = []

        self.steps = steps
        self.proportion_cls = proportion_cls

        # unification of the list of arteries considered in the study
        if artery_type == "LCA":
            self.artery_type = ["LCA"]
        elif artery_type == "RCA":
            self.artery_type = ["RCA"]
        elif artery_type == "LCA/RCA":
            self.artery_type = ["LCA", "RCA"]
        else:
            raise TypeError(f"Wrong artery type, {artery_type}")
        self.slice_selection_params = copy.deepcopy(slice_selection_params)
        self.device = device
        self.max_len = max_len

        # dictionary, where key is the patient number, values are the list of frames belonging to it
        self.studyid_ordernum_dict = {}

        # counters for statistics
        self.all_pictures = 0
        self.good_pictures = 0
        self.bad_pictures = 0

        self.n_first_frames_if_empty = n_first_frames_if_empty
        self.ssim_if_empty = ssim_if_empty

        # if we need to use the model for slice selection, load it
        if self.slice_selection_params['frames_selection_method'] == "model":
            self.slice_selection_model = torchvision.models.convnext_tiny(weights=None, num_classes=2)
            self.slice_selection_model.load_state_dict(torch.load(slice_selection_params['model_slice_selection_path'], map_location=device))
            self.slice_selection_model.to(device)
            self.slice_selection_model.train(False)

        # dataset loading and processing
        self.load_data(input_path)
        self.transform = transform

        # output intermediate statistics
        print(f"Raw pictures: {self.all_pictures}, good: {self.good_pictures}, bad:{self.bad_pictures}, proportion:{self.good_pictures/self.all_pictures}")

        self.seed = seed

        # deleting the model and clearing the cache
        del self.slice_selection_model
        gc.collect()
        torch.cuda.empty_cache() 

    def load_data(self, input_path: str) -> None:
        '''
        Loading and processing data for the dataset

        input_path - path to the dataset folder
        
        return: None
        '''

        # the path to the folder with the left heart type
        if os.path.exists(os.path.join(input_path, "Left_Dominance")):
            left_type = os.path.join(input_path, "Left_Dominance")  
            print("Left_Dominance")

            # loading data for left dominance
            self.read_one_type(left_type)

        # path to the folder with the right heart type
        if os.path.exists(os.path.join(input_path, "Right_Dominance")):
            right_type = os.path.join(input_path, "Right_Dominance") 
            print("Right_Dominance")   

            # loading data for right dominance
            self.read_one_type(right_type)     
    
    def read_one_type(self, path_to_type: str) -> None:
        '''
        Function to read one type of hearts

        path_to_type - path to the folder with a certain type 
        
        return: None
        '''

        # dominance type in abbreviated form
        heart_type = os.path.basename(path_to_type).lower()[:4]

        # patient folders
        study_folders = os.listdir(path_to_type)

        for study in tqdm(study_folders):
            study_path = os.path.join(path_to_type, study)

            for artery_type in os.listdir(study_path):
                # ignore the types of arteries that need to be skipped
                if artery_type not in self.artery_type:
                    continue
                
                # path to the folder with projections of a certain artery type 
                artery_path = os.path.join(study_path, artery_type)

                files = os.listdir(artery_path)

                # go through all the npz
                for file in files:
                    
                    # receiving information for npz
                    file_path = os.path.join(artery_path, file)
                    series = os.path.basename(file_path).replace(".npz", "")
                    hearth_npz_info = np.load(file_path)
                    heart_projection = hearth_npz_info["pixel_array"]
                    # initial_dcm_path = hearth_npz_info["pixel_path"]

                    # filling dictionaries
                    if series not in self.series_dict.values():
                        self.series_dict[self.series_uniq_num] = series
                        cur_series_num = self.series_uniq_num
                        self.series_uniq_num += 1
                    else:
                        cur_series_num = list(self.series_dict.keys())[list(self.series_dict.values()).index(series)]

                    study_num = study.split("_")[-1]
                    
                    # checking that we are using the video and not the frame
                    if len(heart_projection.shape) == 2:
                        heart_projection = np.expand_dims(heart_projection, 0)

                    # counting statistics
                    self.all_pictures += len(heart_projection)
                    self.bad_pictures += len(heart_projection)

                    if self.slice_selection_params['frames_selection_method'] == "model":

                        # selection of slices using the model
                        n_repeat = 1
                        conf_thr = self.slice_selection_params['conf_thr_slices'] 
                        step = 1
                        preds = predict_good_slices(heart_projection, model=self.slice_selection_model, device=self.device)
                        good_i = get_good_idx(preds, conf_thr, **self.slice_selection_params)

                        # if necessary, apply the n_first_frames method if the model has filtered all frames
                        if good_i == [] and self.n_first_frames_if_empty:
                            good_i = list(range(len(heart_projection)))
                            good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']:]
                            heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']:]

                        # if necessary apply ssim method if the model has filtered all frames
                        elif good_i == [] and self.ssim_if_empty:
                            print(len(heart_projection), study)
                            good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                            heart_projection = heart_projection[good_i]
                        else:
                            heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "ssim":
                        # selection of slices using structural similarity
                        self.slice_selection_params["window"] = self.slice_selection_params["window"] - 1

                        good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=self.slice_selection_params)
                        heart_projection = heart_projection[good_i]
                    
                    elif self.slice_selection_params['frames_selection_method'] == "first_frames" and \
                        self.slice_selection_params['n_first_frames_to_skip'] > 0:
                        # selection of slices by discarding the first n slices
                        good_i = list(range(len(heart_projection)))
                        good_i = good_i[self.slice_selection_params['n_first_frames_to_skip']: ]
                        heart_projection = heart_projection[self.slice_selection_params['n_first_frames_to_skip']: ]
                    
                    # without taking slices
                    elif self.slice_selection_params['frames_selection_method'] is None:
                        good_i = list(range(len(heart_projection)))

                    else:
                        raise AssertionError(f"Wrong frames selection method: {self.slice_selection_params['frames_selection_method']}")

                    # counting statistics
                    self.good_pictures += len(good_i)
                    self.bad_pictures -= len(good_i)

                    # checking that we are using the video and not the frame
                    assert len(heart_projection.shape) == 3

                    n_repeat = 1 

                    # adding information for each frame to the dataset
                    for j in range(n_repeat):
                        
                        for i, frame_num in enumerate(good_i):  
            
                            project_dict = {"dominance_type": heart_type, "artery_type": artery_type,
                                            "image": heart_projection[i], 'frame_num': frame_num,  
                                            'studyid': study_num, 'seriesid': series,
                                            'series_num': cur_series_num}
                            
                            # filling in relevant dictionaries and counting statistics 
                            if study not in self.study_series_dict.keys():
                                self.study_series_dict[study] = [series]
                            else:
                                self.study_series_dict[study].append(series)

                            if study not in self.studyid_ordernum_dict.keys():
                                self.studyid_ordernum_dict[study] = [len(self.projection_list)]
                            else:
                                self.studyid_ordernum_dict[study].append(len(self.projection_list))
                            self.projection_list.append(project_dict)

                            # stop when the dataset size is exceeded
                            if self.max_len is not None and len(self.projection_list) >= self.max_len:
                                return


    def __len__(self) -> int:
        return len(self.projection_list)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
                                                           Tuple[torch.Tensor, torch.Tensor, int]]:
        
        # receiving a frame with the appropriate transformations and labels
        proj_i = self.projection_list[idx]  
        
        img = proj_i['image']
        img = np.expand_dims(img, -1)
        img = np.concatenate([img, img, img], -1)
        img = np.array(img, dtype=np.uint8)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img / 255

        # if we test the pipelines, we return labels for occlusion and dominance
        if self.returns_type == "pipeline":

            # assume there are no patients with occlusion in the dataset
            label_occlusion = torch.tensor(0).type(torch.float)
            
            if proj_i['dominance_type'] == 'left':
                label_dominance = torch.tensor(0).type(torch.float)
            elif proj_i['dominance_type'] == 'righ':
                label_dominance = torch.tensor(1).type(torch.float)
            return img, label_occlusion, label_dominance, proj_i['series_num']

        # return labels depending on the current occlusion or dominance classification task
        else:

            if self.label_name == "occlusion":
                if proj_i['is_occlusion'] == False:
                    label = torch.tensor(0).type(torch.float)
                elif proj_i['is_occlusion'] == True:
                    label = torch.tensor(1).type(torch.float)
            elif self.label_name == "artifact":
                if proj_i['is_artifact'] == False:
                    label = torch.tensor(0).type(torch.float)
                elif proj_i['is_artifact'] == True:
                    label = torch.tensor(1).type(torch.float)
            else:
                raise AssertionError(f"Wrong label:{self.label_name}")

        return img, label, proj_i['series_num'] 
    

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
        self.model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights)

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

        self.y_val.clear()
        self.p_val.clear()
        self.r_val.clear()
        

    def on_train_epoch_end(self) -> None:

        # logging at the end of the training epoch
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Any:

        # optimizer parameters for model training
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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