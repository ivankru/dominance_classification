import numpy as np
import torch
from torch.nn import Softmax
import os
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score
from typing import List, Any, Tuple

from utils.calculate_metrics import calculate_metrics
from utils.slices_selection import predict_good_slices, get_good_idx, predict_good_slices_ssim

class trainer():
    '''
    Class for model training
    '''
    def __init__(self, model: Any, optimizer: Any, loss_fn: Any, train_dataloader: Any,
                 val_dataloader: Any, device: str, fold: Any, conf_thr: float) -> None:
        '''
        model - model to be trained
        optimizer - model optimizer
        loss_fn - loss function
        train_dataloader - training dataloader
        val_dataloader - validation dataloader
        device
        fold - fold number
        conf_thr - model confidence threshold for attributing a slice to left/right circulation type

        return: None
        '''

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.fold = fold
        self.conf_thr = conf_thr

        
    def train_one_epoch(self) -> Tuple[float, float, float, float]:
        '''
        One epoch training

        return: loss and metrics values
        '''

        running_loss = 0.
        
        # lists where all the predictions and labels will be stored
        y_true_list = []
        y_pred_list = []
        
        for i, data in (enumerate(self.train_dataloader)):
            # load the data
            inputs, labels, file_paths = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # reset the gradients
            self.optimizer.zero_grad()

            # predictions on the batch
            outputs = self.model(inputs)
            if str(self.model).lower().startswith('convnextv2forimageclassification'):
                outputs = outputs['logits']

            outputs = outputs.to(self.device)
            # calculate loss and gradients
            loss = self.loss_fn(outputs.type(torch.float), labels.type(torch.long))
            loss.backward()

            # take a step
            self.optimizer.step()

            # add loss
            running_loss += loss.item()
            
            # represent the outputs as probabilities
            y_pred = Softmax(dim=1)(outputs)
            
            # represent predictions and labels as np.array
            y_pred = y_pred.cpu().detach().numpy()
            y_true = np.array(labels.cpu().detach().numpy(), dtype=np.int16)
            
            # add them to the corresponding arrays
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            
        # convert all predictions and classes into a single array
        y_pred_arr = np.concatenate(y_pred_list)
        y_true_arr = np.concatenate(y_true_list)
        
        y_pred_arr_labels = np.array(y_pred_arr > self.conf_thr, dtype=int)

        # calculate average loss and roc_auc
        avg_loss = running_loss / (i + 1)
        try:
            auc = roc_auc_score(y_true_arr, y_pred_arr[:, 1])
        except:
            print("The roc-auc metric is undefined in the case of a single class and will be equated to 0")
            auc = 0.0
        
        f1 = f1_score(y_true_arr, y_pred_arr_labels[:, 1], average='macro')
        
        mcc = matthews_corrcoef(y_true_arr, y_pred_arr_labels[:, 1])
        
        return avg_loss, auc, f1, mcc

    def validate_one_epoch(self) -> Tuple[float, float, float, float]:
        '''
        Validation of one epoch

        return: loss and metrics values
        '''
        
        running_loss = 0.
        
        # lists where all the predictions and labels will lie
        y_true_list = []
        y_pred_list = []
        
        for i, data in (enumerate(self.val_dataloader)):
            # load the data
            inputs, labels, file_paths = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # predictions on the batch
            outputs = self.model(inputs)
            if str(self.model).lower().startswith('convnextv2forimageclassification'):
                outputs = outputs['logits']
            outputs = outputs.to(self.device)
            # count the loss
            loss = self.loss_fn(outputs.type(torch.float), labels.type(torch.long))
            
            # add loss
            running_loss += loss.item()
            
            # represent the outputs as probabilities
            y_pred = Softmax(dim=1)(outputs)
            
            # represent predictions and labels as np.array
            y_pred = y_pred.cpu().detach().numpy()
            y_true = np.array(labels.cpu().detach().numpy(), dtype=np.int16)
            
            # add them to the corresponding arrays
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            
        # convert all predictions and classes into a single array
        y_pred_arr = np.concatenate(y_pred_list)
        y_true_arr = np.concatenate(y_true_list)
        
        y_pred_arr_labels = np.array(y_pred_arr > self.conf_thr, dtype=int)

        # calculate average loss and roc_auc
        avg_loss = running_loss / (i + 1)
        try:
            auc = roc_auc_score(y_true_arr, y_pred_arr[:, 1])
        except:
            print("The roc-auc metric is undefined in the case of a single class and will be equated to 0")
            auc = 0.0
        
        f1 = f1_score(y_true_arr, y_pred_arr_labels[:, 1], average='macro')
        
        mcc = matthews_corrcoef(y_true_arr, y_pred_arr_labels[:, 1])

        return avg_loss, auc, f1, mcc

    def train_n_epochs(self, epochs: int, models_folder: str) -> Tuple[float, float, float]:
        ''' 
        Function of training n epochs

        epochs - number of epochs
        models_folder - folder where to save models

        return: best validation loss and metrics
        '''
        epoch_number = 0

        best_vloss = 1000
        best_vauc = 0
        best_vf1 = 0

        for epoch in range(0, epochs):
            # print(torch.cuda.memory_allocated(device=1))
            print()

            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] = 1e-5 
            elif (epoch >= 4) and (str(self.model).lower().startswith('swintransformer')):
                 self.optimizer.param_groups[0]['lr'] = 1e-4 * .5 
            else:
                self.optimizer.param_groups[0]['lr'] = 1e-4 
            print(f'EPOCH {epoch}:')

            # training
            self.model.train(True)
            avg_loss, auc, f1, mcc = self.train_one_epoch()

            # validation
            self.model.train(False)
            with torch.no_grad():
                avg_vloss, vauc, vf1, vmcc = self.validate_one_epoch()

            print(f'LOSS train {avg_loss} valid {avg_vloss}')
            print(f'AUC train {auc} valid {vauc}')
            print(f'F1 train {f1} valid {vf1}')
            print(f'MCC train {mcc} valid {vmcc}')

            # wandb.log({'loss train': avg_loss, 'loss val': avg_vloss, 'auc train': auc, 'auc val': vauc, 
            #            'f1 train': f1, 'f1 val': vf1})

            if vauc > best_vauc :
                best_vauc = vauc

                model_path = os.path.join(models_folder, f'heart_model_{self.fold}_{np.round(vauc, 3)}_{np.round(vf1, 3)}_{np.round(avg_vloss, 3)}.pt')
                torch.save(self.model.state_dict(), model_path)

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss

                model_path = os.path.join(models_folder, f'heart_model_{self.fold}_{np.round(vauc, 3)}_{np.round(vf1, 3)}_{np.round(avg_vloss, 3)}.pt')
                torch.save(self.model.state_dict(), model_path)

            if vf1 > best_vf1 :
                best_vf1 = vf1

                model_path = os.path.join(models_folder, f'heart_model_{self.fold}_{np.round(vauc, 3)}_{np.round(vf1, 3)}_{np.round(avg_vloss, 3)}.pt')
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
            
        return best_vauc, best_vf1, best_vloss

def eval_nn(model: Any, test_dataloader: Any, device: str,
            loss_fn: Any, conf_thr: float) -> Any: 
    '''
    Function for estimating a single model
    
    model - model to be evaluated
    test_dataloader - test dataloader
    device - device
    loss_fn - test loss function
    conf_thr - model confidence threshold

    return: metrics, bad images and unsure predictions
    '''

    model.train(False)
    with torch.no_grad():
        running_loss = 0.

        # lists where all the predictions and labels will be stored 
        y_true_list = []
        y_pred_list = []
        
        y_pred_bad = []
        y_true_bad = []
        images_bad = []
        
        y_pred_unsure = []
        y_true_unsure = []
        images_unsure = []
        
        file_paths_bad = []
        
        for i, data in (enumerate(test_dataloader)):
            # load the data
            inputs, labels, file_paths = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # predictions on the batch
            outputs = model(inputs)

            if str(model).lower().startswith('convnextv2forimageclassification'):
                outputs = outputs['logits']

            outputs = outputs.to(device)
            # count the loss
            loss = loss_fn(outputs.type(torch.float), labels.type(torch.long))

            # add loss
            running_loss += loss.item()

            # represent the outputs as probabilities
            y_pred = Softmax(dim=1)(outputs)

            # represent predictions and labels as np.array
            y_pred = y_pred.cpu().detach().numpy()
            y_true = np.array(labels.cpu().detach().numpy(), dtype=np.int16)
            
            for j in range(len(y_true)):
                if (y_pred[j, 1] > 0.5) != (y_true[j] > 0.5):
                    y_pred_bad.append(y_pred[j, 1])
                    y_true_bad.append(y_true[j])
                    images_bad.append(inputs.cpu().detach().numpy()[j, 0])
                    file_paths_bad.append(file_paths[j])
                if 0.35 < y_pred[j, 1] < 0.65:
                    y_pred_unsure.append(y_pred[j, 1])
                    y_true_unsure.append(y_true[j])
                    images_unsure.append(inputs.cpu().detach().numpy()[j, 0])
            
            # add them to the corresponding arrays
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)

        # convert all predictions and classes into a single array
        y_pred_arr = np.concatenate(y_pred_list)
        y_true_arr = np.concatenate(y_true_list)

        y_pred_arr_labels = np.array(y_pred_arr > conf_thr, dtype=int)
        # calculate average loss and roc_auc
        avg_loss = running_loss / (i + 1)
        
        try:
            auc = roc_auc_score(y_true_arr, y_pred_arr[:, 1])
        except:
            print("The roc-auc metric is undefined in the case of a single class and will be equated to 0")
            auc = 0.0

        f1 = f1_score(y_true_arr, y_pred_arr.argmax(axis=1), average='macro')
        
        dict_metrics = calculate_metrics(y_true_arr, y_pred_arr, metrics=[recall_score, precision_score], metrics_general=[matthews_corrcoef], conf_thr=conf_thr)
        dict_metrics['avg_loss'] = [avg_loss]
        dict_metrics['auc'] = [auc]
        dict_metrics['f1'] = [f1]


        images_bad = np.array(images_bad)
        y_true_bad = np.array(y_true_bad)
        y_pred_bad = np.array(y_pred_bad)
        
        y_pred_unsure = np.array(y_pred_unsure)
        y_true_unsure = np.array(y_true_unsure)
        images_unsure = np.array(images_unsure)
        
    return avg_loss, auc, f1, y_true_bad, y_pred_bad, images_bad, file_paths_bad, y_true_unsure, y_pred_unsure, images_unsure, dict_metrics

def predict_one_projection(heart_projection: np.array, file_path: str, model: Any, 
                           slice_selection_model: Any, conf_thr_slices: float,
                           conf_thr: float, device: str, transform: Any,
                           test_slice_selection_params: dict) -> np.array:  
    '''
    heart_projection - projection to be processed
    file_path - path to the projection file
    model - model that will predict the type of blood circulation on the slice
    slice_selection_model - model for slice selection on the projection
    conf_thr_slices - confidence threshold for deciding that a slice is bad/good 
    conf_thr - confidence threshold for deciding which type of blood circulation the slice belongs to
    device
    transform
    test_slice_selection_params - slice selection parameters (bad/good)

    return: prediction for one projection
    '''

    preds = predict_good_slices(heart_projection, model=slice_selection_model, device=device, **test_slice_selection_params)
    
    if 'use_model' in test_slice_selection_params.keys():
        if test_slice_selection_params['use_model'] == False:
            conf_thr_slices = -1
    else:
        pass
    
    good_i = get_good_idx(preds, conf_thr_slices, **test_slice_selection_params)
    heart_projection = heart_projection[good_i]

    if len(good_i) == 0:
        print(f'No suitable slices')
        print(file_path)
        return -1
    
    heart_projection = np.expand_dims(heart_projection, -1)
    heart_projection = np.concatenate([heart_projection] * 3, -1)
    heart_projection = np.array(heart_projection, dtype=np.uint8)

    print(heart_projection.shape)
    heart_projection_arr = []
    for i in range(len(heart_projection)):
        augmented = transform(image=heart_projection[i])
        img = augmented['image']
        heart_projection_arr.append(img)

    
    heart_projection = torch.stack(heart_projection_arr)
    heart_projection = heart_projection / 255

    print(heart_projection.shape, heart_projection.min(), heart_projection.max())

    heart_projection = heart_projection.to(device)
    
    preds = model(heart_projection)
    preds = Softmax(dim=1)(preds)[:, 1].cpu().detach().numpy()

    return np.mean(preds > conf_thr)
    # return np.mean(preds)

def predict_one_projection_ssim(heart_projection: np.array, file_path: str, model: Any, conf_thr: float, 
                                device: str, transform: Any, test_slice_selection_params: dict) -> np.array:  
    '''
    heart_projection - projection to be processed
    file_path - path to the projection file
    model - model that will predict the type of blood circulation on the slice
    conf_thr - confidence threshold for deciding which circulation type the slice belongs to.
    device
    transform
    test_slice_selection_params - slice selection parameters (bad/good)

    return: prediction for one projection
    '''

    good_i = predict_good_slices_ssim(heart_projection, slice_selection_params=test_slice_selection_params)
    heart_projection = heart_projection[good_i]

    if len(good_i) == 0:
        print(f'No suitable slices')
        print(file_path)
        return -1
    
    heart_projection = np.expand_dims(heart_projection, -1)
    heart_projection = np.concatenate([heart_projection] * 3, -1)
    heart_projection = np.array(heart_projection, dtype=np.uint8)

    print(heart_projection.shape)
    heart_projection_arr = []
    for i in range(len(heart_projection)):
        augmented = transform(image=heart_projection[i])
        img = augmented['image']
        heart_projection_arr.append(img)

    
    heart_projection = torch.stack(heart_projection_arr)
    heart_projection = heart_projection / 255

    print(heart_projection.shape, heart_projection.min(), heart_projection.max())

    heart_projection = heart_projection.to(device)
    
    preds = model(heart_projection)
    preds = Softmax(dim=1)(preds)[:, 1].cpu().detach().numpy()

    return np.mean(preds > conf_thr)
    # return np.mean(preds)