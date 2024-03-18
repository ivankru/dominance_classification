import numpy as np
import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from typing import List, Any

def get_good_idx_1(preds: Any, conf_thr: float, window: int, **kwargs) -> List[int]:
    '''
    Takes as input a list of predictions for each slice, returns, based on the selected conditions 
    indices of slices that have slices suitable for classifying the artery type (range from i to j)

    preds - list with predictions
    window - width of the window in which each prediction must be above the probability threshold
    conf_thr - probability threshold

    return: list of the indexes of the good slices
    '''

    good_counter = 0
    start = 0
    end = 0
    
    for i in range(len(preds) - 1, -1, -1):
        if preds[i] >= conf_thr:
            good_counter += 1
        elif preds[i] < conf_thr:
            good_counter = 0
        
        if good_counter == window:
            end = i + window
            break
    
    good_counter = 0
    for i in range(0, len(preds)):
        if preds[i] >= conf_thr:
            good_counter += 1
        elif preds[i] < conf_thr:
            good_counter = 0
        
        if good_counter == window:
            start = i - window + 1
            break
        
    good_i = [i for i in range(start, end)]
    return good_i

def get_good_idx_2(preds: Any, conf_thr: float, window: int, **kwargs) -> List[int]:
    '''
    Takes as input a list of predictions for each slice, based on a moving average
    returns the slice indices of slices that have slices suitable for classifying the artery type

    preds - list with predictions
    window - width of the window for calculating the moving average
    conf_thr - probability threshold

    return: list of the indexes of the good slices
    '''

    half_window = int(window / 2)
    good_i = []
    for i in range(half_window, len(preds) - half_window):
        if 0.25 * preds[i - 1] + 0.5 * preds[i] + 0.25 * preds[i + 1] > conf_thr:
            good_i.append(i)
    return good_i

def get_good_idx(preds: Any, conf_thr: float, **kwargs) -> List[int]:
    '''
    function for getting good/bad slices
    preds - model predictions to classify slices into good/bad slices
    conf_thr - threshold for selecting good/bad slices
    kwargs - additional arguments

    return: list of the indexes of the good slices
    '''
    
    method = kwargs['method_num']
    window = kwargs['window']

    if method == 1:
        
        good_i = get_good_idx_1(preds, conf_thr, window)
    elif method == 2:
        good_i = get_good_idx_2(preds, conf_thr, window)
    else:
        print('Wrong method')
        good_i = []

    return good_i
    
def predict_good_slices(heart_projection: np.array, model: Any, device: str, **kwargs) -> np.array:
    '''
    Takes a projection as input, returns start and end indices

    heart_projection - heart projection
    model - the model on which the prediction will be performed
    device - what the prediction will be performed on
    window - width of the window in which each prediction must be above the probability threshold
    conf_thr - probability threshold 

    return: prediction about is the slices good
    '''
    batch_size = 100

    transform = transforms.Compose([transforms.Resize(224, antialias=False)])

    heart_projection_slice = np.expand_dims(heart_projection, 1)
    heart_projection_slice = np.concatenate([heart_projection_slice] * 3, 1)
    heart_projection_slice = np.array(heart_projection_slice, dtype='float32')
    heart_projection_slice = torch.from_numpy(heart_projection_slice)
    heart_projection_slice = heart_projection_slice.type(torch.uint8)
    heart_projection_slice = transform(heart_projection_slice)
    heart_projection_slice = heart_projection_slice / 255

    # general_preds = []

    # for i in range(math.ceil(len(heart_projection_slice) / batch_size)):
    #     heart_projection_slice_tmp = heart_projection_slice[i * batch_size: (i + 1) * batch_size].to(device)
    #     preds = model(heart_projection_slice_tmp)
    #     preds = torch.nn.Softmax(dim=1)(preds)

    #     preds = preds[:, 1].cpu().detach().numpy()

    #     general_preds.append(preds)

    # general_preds = np.concatenate(general_preds, axis=0)

    heart_projection_slice = heart_projection_slice.to(device)
    preds = model(heart_projection_slice)
    preds = torch.nn.Softmax(dim=1)(preds)

    preds = preds[:, 1].cpu().detach().numpy()
    
    return preds

def predict_good_slices_ssim(heart_projection: np.array, slice_selection_params: dict,
                              **kwargs) -> np.array:
    ssim_score_arr = []

    for i in range(0, len(heart_projection)):
        img_0 = heart_projection[0]
        img_i = heart_projection[i]
        
        ssim_score = ssim(img_0, img_i, data_range=max([img_i.max(), img_0.max()]) - min([img_i.min(), img_0.min()]))
        ssim_score_arr.append(ssim_score)
        
    ssim_score_arr = np.array(ssim_score_arr)

    peak_idx = np.argmin(ssim_score_arr)

    window = slice_selection_params['window']
    half_window = window // 2

    min_idx = max(0, peak_idx - half_window)
    max_idx = min(len(heart_projection), peak_idx + half_window + 1)

    good_i = np.arange(min_idx, max_idx)

    return good_i



