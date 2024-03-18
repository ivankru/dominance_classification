import numpy as np 
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import os
import torch
from tqdm import tqdm
from typing import List, Union, Tuple, Optional

from utils.slices_selection import get_good_idx, predict_good_slices, predict_good_slices_ssim  

class HeartLR_simple_main(Dataset):
    def __init__(self, input_path: str, features: List[str] = [], device: str = 'cpu', transform: Optional[str] = None,
                 slice_selection_params: Optional[dict] = None, artery_type: Optional[str] = None, steps: List[int] = [1, 1],
                 seed: int = 42, max_len: Optional[int] = None, proportion_cls: List[int] = [1, 1], status: str = "train") -> None:
        '''
        Class for the main dataset for the heart type classification task with new occlusions and left dominances included therein

        input_path - path to the folder with the dataset
        features - considered features
        device - device on which calculations are performed
        transform - transformations
        slice_selection_params - dictionary with instructions for selecting significant frames
        artery_type - artery type under consideration
        steps - list of steps with which we select frames for left and right dominance
        seed - fixing of random transformations
        max_len - maximum dataset size
        proportion_cls - repetitions of each class
        status - determines whether to create a dataset for training or test phase

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

        # main list for the dataset with information for each frame
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
        print("How many Left_Dominance:", self.count_label_0)
        print("How many Right_Dominance:", self.count_label_1)

        self.seed = seed

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
                    if study_num not in self.studies_set:
                        self.studies_set.add(study_num)
                    
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
                            if heart_type == 'left':
                                n_repeat = self.proportion_cls[0]
                                step = self.steps[0]
                                conf_thr = self.slice_selection_params['conf_thr_slices_left']

                            elif heart_type == 'righ':
                                n_repeat = self.proportion_cls[0]
                                step = self.steps[1]
                                conf_thr = self.slice_selection_params['conf_thr_slices_right']

                        elif self.status == "test":
                            conf_thr = self.slice_selection_params['conf_thr_slices'] 
                        else:
                            raise AssertionError(f"Wrong status: {self.status}")
                        
                        preds = predict_good_slices(heart_projection, model=self.slice_selection_model, device=self.device)
                        good_i = get_good_idx(preds, conf_thr, **self.slice_selection_params)
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
          
                            project_dict = {"dominance_type": heart_type, "artery_type": artery_type, 'series_num': cur_series_num,\
                                            "image": heart_projection[i], 'frame_num': frame_num, \
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

                            if project_dict['dominance_type'] == 'left':
                                self.count_label_0 += 1
                            elif project_dict['dominance_type'] == 'righ':
                                self.count_label_1 += 1
                            else:
                                raise AssertionError(f"Wrong dominance type: {project_dict['dominance_type']}")

                            # stop when the dataset size is exceeded
                            if self.max_len is not None and len(self.projection_list) >= self.max_len:
                                return

    def __len__(self) -> int:
        return len(self.projection_list)
    
    def __getitem__(self, idx: Union[int, slice]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
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

        if proj_i['dominance_type'] == 'left':
            label = torch.tensor(0).type(torch.float)
        elif proj_i['dominance_type'] == 'righ':
            label = torch.tensor(1).type(torch.float)

        return img, label, proj_i['series_num'] 
