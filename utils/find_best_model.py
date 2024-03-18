import os
from typing import Tuple

def find_best_models(folder: str, fold_num: int) -> Tuple[str, str, str]:
    '''
    function for selecting the best models by loss, roc_auc, f1 criteria obtained on validation datasets
    folder - path to the folder where the models are located
    fold_num - number of the fold, models to be selected for

    return: paths to the best models
    '''
    
    models = [model for model in os.listdir(folder) if model.lower().startswith('heart')]
            
    models = [[model] + model.replace('.pt', '').split('_')[-3:] for model in models if int(model.split('_')[2]) == fold_num]
    models = list(map(lambda x: [x[0], float(x[1]), float(x[2]), float(x[3])], models))

    best_model_auc = sorted(models, key=lambda x: [x[1], -x[3], x[2]])[-1][0]
    best_model_f1 = sorted(models, key=lambda x: [x[2], -x[3], x[1]])[-1][0]
    best_model_loss = sorted(models, key=lambda x: [-x[3], x[1], x[2]])[-1][0]

    print(*[best_model_auc, best_model_f1, best_model_loss])

    for model in os.listdir(folder):
        # if (model not in [best_model_loss, best_model_auc, best_model_f1]) and (int(model.split('_')[2]) == fold_num):
        if (model not in [best_model_loss]) and (int(model.split('_')[2]) == fold_num):
            model_path = os.path.join(folder, model)
            os.remove(model_path)

    return best_model_loss, best_model_auc, best_model_f1
    # return best_model_loss
