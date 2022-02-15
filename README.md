# DEEP VIDEO INPAINTING LOCALIZATION USING SPATIAL AND TEMPORAL TRACES



## Requirements
- Python 3
- Tensorflow >= 1.10.0


## Usage
### Train
* First, prepare the training data so that the images are stored in "xxx/png/xxx/" and the corresponding groundtruth masks are stored in "xxx/msk/xxx/".

* Second, generate the flows by RAFT(https://github.com/princeton-vl/RAFT), so that the forward flows are stored in "xxx/forward/xxx" and the backward flows are stored in "xxx/backward/xxx". 

Then, run the following command.
### Train
```
python3 hp_fcn.py --data_dir <path_to_the_training_dataset> --logdir <path_to_the directory_for_saving_model_and_log> --mode train
```
for example,
```
python3 hp_fcn.py --data_dir example_data/png/* --logdir save/model --mode train
```

### Test
Prepare the testing data in a similar way and run the code as follows.
```
python3 hp_fcn.py --data_dir <path_to_the_testing_dataset> --logdir <path_to_the directory_where_the_trained_model_is_saved> --mode test
```

 

