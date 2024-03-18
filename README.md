# __Dominance classification__

The 2d models convnext, swin, convnextv2 and 3d models R3D_18, Mednext are used for classification.

### __Project structure__

- configs - folder with configs. Each config file has a unique experiment number. Training configs contain information about dataset parameters, folders for saving models and results, training parameters. Test configs contain information about folders with models for testing and dataset parameters

- datasets - folder with datasets for research

- logs - folder for saving results

- logs_tensorboard - folder for saving logs for tensorboard

- models - folder for saving models

- train_dominance - folder with scripts for training dominance classification model by frames

  ```bash
  train_dominance
    |-- dataset_class.py
    |-- train.py
    |-- train.sh
    |-- transforms.py
  ```

    - dataset_class.py - script for initializing classes with dataset
    
    - train.py - script for training dominance classification models by frames
    
    - train.sh - script to run the training
    
    - transforms.py - script for initialization of functions with transforms

- train_dominance_video_test_pipeline - folder with scripts for training occlusion classification model by frames and testing of pipeline

  ```bash
  train_occlusion_test_pipeline
    |-- dataset_class_video.py
    |-- mednext_blocks.py
    |-- mednext.py
    |-- test_full_pipeline_on_domain_shift_or_special_feature.py
    |-- test_full_pipeline_on_folds.py
    |-- test_full_pipeline_on_hold_out.py
    |-- test.sh
    |-- train.py
    |-- train.sh
  ```
    
    - dataset_class_video.py - script for initializing classes with dataset
    
    - mednext_blocks.py - script for initializing Mednext model blocks
    
    - mednext.py - script for Mednext model initialization
    
    - test_full_pipeline_on_domain_shift_or_special_feature.py - script for testing pipeline with models of occlusion classification by frames and dominance by video on domain shift or folders with patient features
    
    - test_full_pipeline_on_folds.py - script for testing the pipeline with frame-based occlusion classification and video dominance models on test parts of the main dataset partitions
    
    - test_full_pipeline_on_hold_out.py - script for testing the pipeline with models of occlusion classification by frames and dominance by video on hold out
    
    - test.sh - script to start testing
    
    - train.py - script for training occlusion and dominance classification models on video 
    
    - train.sh - script to run training

- train_occlusion_test_pipeline - folder with scripts for training the dominance classification model by video and testing the pipeline

  ```bash
  train_occlusion_test_pipeline
    |-- dataset_class.py
    |-- test_full_pipeline_on_domain_shift_or_special_feature.py
    |-- test_full_pipeline_on_folds.py
    |-- test_full_pipeline_on_hold_out.py
    |-- test.sh
    |-- train.py
    |-- train.sh
    |-- transforms.py
  ```
    
    - dataset_class.py - script for initializing classes with dataset 
    
    - test_full_pipeline_on_domain_shift_or_special_feature.py - script for testing a pipeline with occlusion and dominance classification models by frames on domain shift or patient feature folders
    
    - test_full_pipeline_on_folds.py - script for testing the pipeline with occlusion and frame dominance classification models on test parts of the main dataset partitions
    
    - test_full_pipeline_on_hold_out.py - script for testing the pipeline with occlusion and frame dominance classification models on hold out
    
    - test.sh - script to start testing
    
    - train.py - script for training occlusion classification models
    
    - train.sh - script to run training
    
    - transforms.py - script for initializing functions with transforms

- utils - utils

  ```bash
  utils
    |-- __init__.py
    |-- calculate_metrics.py
    |-- find_best_model.py
    |-- loss.py
    |-- parse_config.py
    |-- set_seed.py
    |-- slices_selection.py
    |-- train_test_functions.py
    |-- wandb_scripts.py
  ```

- \_\_\_init__.py - module initialization file

  - calculate_metrics.py - file for calculating metrics

  - find_best_model.py - script for finding the best models

  - loss.py - file with implementation of loss functions

  - parse_configs.py - script for parsing parameters from config file 

  - set_seed.py - script for setting sids

  - slice_selection.py - functions for slice selection

  - train_test_functions.py - functions for training and testing models

### __Installation__
```bash
pipenv sync 
pipenv shell
```

### __Training__
For training and testing of cross-validation models it is necessary to create a configuration file in the format train_cross_val_config.yaml in the folder configs/exp_1 (select the desired experiment number). The obtained models and logs will be saved in the directory specified in the config in the folder of the corresponding experiment.

To start training the dominance classification model by frames as parameters in the train.sh script, specify the experiment number and the flag (whether augmentations are needed) in the script and execute the command:

```bash
bash train_dominance/train.sh
```

To run the training of the frame-based occlusion classification model, the experiment number and the flag (whether augmentations are needed) in the train.sh script must be specified as parameters in the script and the command must be executed:

```bash
bash train_occlusion_test_pipeline/train.sh
```

To run the training of a dominance or occlusion classification model from video as parameters in the train.sh script, you must specify the experiment number in the script and execute the command:

```bash
bash train_dominance_video_test_pipeline/train.sh
```

### __Testing__
To start testing it is necessary to create a configuration file in the format train_cross_val_config.yaml in the folder configs/exp_1 (select the required experiment number).

To test the pipline with the model of dominance classification by frames by test parts of partitions in the script test.sh, it is necessary to uncomment the corresponding line and as parameters it is necessary to specify the number of the experiment and the necessary values for flags to use the method ssim or n_first_frames in the absence of good frames and execute the command:

```bash
bash train_occlusion_test_pipeline/test.sh
```

To test the pipline with the model of dominance classification by frames by hold out in the script test.sh uncomment the corresponding line and as parameters it is necessary to specify the number of the experiment and necessary values for flags to use the method ssim or n_first_frames in case of absence of good frames and execute the command:

```bash
bash train_occlusion_test_pipeline/test.sh
```

To test the pipline with the model of dominance classification by domain shift frames or folders with features in the main dataset, the corresponding line in the test.sh script should be uncommented and the parameters should include the experiment number, the desired value for the is_domain_shift_dataset flag and for the flags to use the ssim or n_first_frames method when there are no good frames and execute the command:

```bash
bash train_occlusion_test_pipeline/test.sh
```

To test the pipeline with the video dominance classification model on test parts of partitions in the script test.sh, the corresponding line should be uncommented and as parameters you should specify the experiment number and the necessary values for flags to use the ssim or n_first_frames method when there are no good frames and execute the command:

```bash
bash train_dominance_video_test_pipeline/test.sh
```

To test the pipeline with the video dominance classification model by hold out in the test.sh script, the corresponding line should be uncommented and the experiment number and the required values for the flags to use the ssim or n_first_frames method when there are no good frames should be specified as parameters and the command should be executed:

```bash
bash train_dominance_video_test_pipeline/test.sh
```

To test the pipeline with the dominance classification model on video by domain shift or feature folders in the main dataset in the test.sh script, the corresponding line should be uncommented and as parameters you should specify the experiment number, the desired value for the is_domain_shift_dataset flag and for the flags to use the ssim or n_first_frames method when there are no good frames and execute the command:

```bash
bash train_dominance_video_test_pipeline/test.sh
```

### __Cofiguration file__

Sample config folders for all the experiments presented above on a small dataset are in the configs folder