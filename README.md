# detectionjs

To use this project:

run:

npm install in the root directory to get the required packages.

To train a model:
$ cd src
$ node trainModel.js --dataset=chess2 --batchSize=1 --epochs=10

There are 2 datasets included with this project. Both of them contain the same images but different annotations. These datasets are chess and chess and they can be found under the datasets directory.

We would recommend using batch sizes smaller than 8 as the model's parameters take up a lot of memory already. However if this is run on the CPU then there is no need.

We encountered some problems while training on the cpu, so in case this happens to you, we alreaady added pretrained versions of our best models on each dataset.
Those models can be found under the models directory in each folder.

To run the evaluation script:

cd third_party/pascalvoceval
$ pip install -r 'requirements.txt'
$ pyhton _init_paths.py

To evaluate on "chess" Dataset on IOU thresh of 0.5
$ python pascalvoc.py -gt ..\..\datasets\chess\test -det ..\..\datasets\chess\models\iteration11\test_predictions --noplot -t 0.5 -sp ..\..\datasets\chess\models\iteration11\evaluation


To evaluate on "chess2" Dataset on IOU thresh of 0.25

python pascalvoc.py -gt ..\..\datasets\chess2\test -det ..\..\datasets\chess2\models\iteration3\test_predictions --noplot -t 0.25 -sp ..\..\datasets\chess2\models\iteration3\evaluation@25

If you trained a version of the model successfully, make sure to add the the evaluation directories under the model (i.e evaluation@25 , evaluation, evaluation@75). Otherwise the script can not save the plots. Also make sure to change the iteration number in the paths above. 

You can also evaluate on training and validation datasests 
:

python pascalvoc.py -gt ..\..\datasets\chess2\valid -det ..\..\datasets\chess2\models\iteration3\valid_predictions --noplot -t 0.25 -sp ..\..\datasets\chess2\models\iteration3\evaluation@25


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


