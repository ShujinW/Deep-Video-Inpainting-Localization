# DEEP VIDEO INPAINTING LOCALIZATION USING SPATIAL AND TEMPORAL TRACES



## Requirements
- Python 3
- Tensorflow >= 1.10.0


## Usage
### Required Data
* First, prepare the training data so that the images are stored in "xxx/png/xxx/" and the corresponding groundtruth masks are stored in "xxx/msk/xxx/".

* Second, generate the flows by RAFT(https://github.com/princeton-vl/RAFT), so that the forward flows are stored in "xxx/forward/xxx" and the backward flows are stored in "xxx/backward/xxx". 

* (Optional) This weight was trained by the FGVC dataset. https://pan.baidu.com/s/1IzoruGkNuCOVYA4iyIZwew (sjyx)

* (Optional) This weight was trained by the STTN dataset. https://pan.baidu.com/s/1pIbZ1j6xDNbi9utnRPCO9w (sjyx)

Then, run the following command.
### Train
```
python3 hp_flow_lstm.py --data_dir <path_to_the_training_dataset> --logdir <path_to_the directory_for_saving_model_and_log> --mode train
```
For example,
```
python3 hp_flow_lstm.py --data_dir example_data/png --logdir model/model --mode train
```

### Test
Prepare the testing data in a similar way and run the code as follows.
```
python3 hp_flow_lstm.py --data_dir <path_to_the_testing_dataset> --logdir <path_to_the directory_where_the_trained_model_is_saved> --mode test
```
For example,
```
python hp_flow_lstm.py --data_dir example_data/png --logdir model/Flow --mode test
```

 

