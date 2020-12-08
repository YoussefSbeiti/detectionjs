# detectionjs

To use this project you need 2 scripts:
One for model training and one for evaluation.

The training script takes one parameter which is the directory containing the dataset. Some dataset directories are included with the project.
If you wish to add a datset of your own, each dataset directory needs to have the following structure:

-- dataset_dir
    |
    |___train
    |       
    |___valid
    |
    |___test
    |
    |___models
    |
    |__ _darknet.labels.txt

where each of the sub directories train,test,and valid contains, for each sample, an image and an annotaion file. 
Important!! Models trained on a dataset get saved in the models directory under that dataset. The models directory needs to be there before training or the model won't be saved.  Please refer to the sample datasets for examples.


