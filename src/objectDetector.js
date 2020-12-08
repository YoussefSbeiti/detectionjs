const { data, layers } = require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs-node-gpu')
const fetch = require('node-fetch');
const {createFunctions} = require("./Yolo")
const fs = require("fs")
const prompt = require('prompt');

var objectDetector = function(modelConfig){

    this.modelConfig = modelConfig
    this.yolo = createFunctions(tf.tensor(modelConfig.anchors), modelConfig.gridSize, modelConfig.numOfClasses)

}

objectDetector.prototype.buildModel = async function(manifestUrl , weightsUrl){
     
    var numOfClasses = this.modelConfig.numOfClasses
    var numOfAnchors = this.modelConfig.anchors.length

    function conv2d(kernelSize, filters,strides,name, trainable = true, preTrained = true,batchInputShape=undefined, useBias = false , padding = "same"){
        if(preTrained){
            var weights = [weightsMap[ name + "/kernel"]]// , weightsMap[name + "/bias"] ];
            console.log("creating preTrained")
            return tf.layers.conv2d({ useBias:useBias, 
                                padding:padding,
                                filters:filters,
                                kernelSize:kernelSize,
                                name:name, 
                                strides:strides,
                                batchInputShape:batchInputShape,
                               // kernelRegularizer: trainable ? tf.regularizers.l2({l2:0.001}) : null,
                                weights: weights,
                                trainable:trainable
                            })    
                        }         
        console.log("Creating non trained")
        return tf.layers.conv2d({ useBias:useBias, 
                                padding:padding,
                                filters:filters,
                                kernelSize:kernelSize,
                                name:name, 
                                strides:strides,
                                batchInputShape:batchInputShape,
                                //kernelRegularizer: tf.regularizers.l2({l2:0.005}),
                                kernelInitializer : tf.initializers.leCunUniform({}),
                                trainable:trainable
                            })    
    }

    var manifest = await fetch(manifestUrl);
    var weightManifest = await manifest.json();
    var weightsMap = await tf.io.loadWeights(weightManifest, weightsUrl);
   
    var finalLayerWeights = weightsMap["m_outputs0/kernel"].reshape([512,5,7])
    finalLayerWeights = finalLayerWeights.slice([0,0,0] , [-1,-1,5])
    finalLayerWeights = finalLayerWeights.concat([tf.fill([512,5,numOfClasses] , 0)] , -1)
    finalLayerWeights = finalLayerWeights.reshape([1,1,512,numOfAnchors*(5+numOfClasses)])

    var finalLayerBiases = weightsMap["m_outputs0/bias"].reshape([5,7])
    finalLayerBiases = finalLayerBiases.slice([0,0] , [-1,5])
    finalLayerBiases = finalLayerBiases.concat([tf.fill([5,numOfClasses] , 0)] , -1)
    finalLayerBiases = finalLayerBiases.reshape([numOfAnchors*(5+numOfClasses)])
    weightsMap["m_outputs0/kernel"] = finalLayerWeights
    weightsMap["m_outputs0/bias"] = finalLayerBiases

    var model = tf.sequential()
    model.add(conv2d(3,16,1,"layer1_conv",true,true, [1,416,416,3]))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,32,1,"layer2_conv",true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,64,1,"layer3_conv",true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,128,1,"layer4_conv",true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    
    model.add(conv2d(3,256,1,"layer5_conv",true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,512,1,"layer6_conv",true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "same" , poolSize: 2 , strides:1}))
    
    model.add(conv2d(3,1024,1,"layer7_conv", true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
   
    model.add(conv2d(3,512,1,"layer8_conv", true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    model.add(tf.layers.leakyReLU({alpha:0.1}))

    model.add(conv2d(1,(5+numOfClasses)*numOfAnchors,1,"m_outputs0", true))
    model.add(tf.layers.batchNormalization({/*betaRegularizer: tf.regularizers.l2({l2:0.001}),gammaRegularizer: tf.regularizers.l2({l2:0.001})*/ } ))
    
    
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: this.yolo.batchLoss,
        metrics : [this.yolo.accuracy(this.yolo.extractBoxesFromPredictedLabelBatch, this.yolo.extractBoxesFromTrueLabelBatch),
            this.yolo.precision(this.yolo.extractBoxesFromPredictedLabelBatch, this.yolo.extractBoxesFromTrueLabelBatch)]
    })
    
    console.log("Model built successfully!")
    console.log(model.summary())
    console.log(tf.memory())

    this.model = model

}

/*
*   Fit the model on the samples stored in the classifier's data Object.
*   @param{int} epochs
*   @param{int} batch size
*/
objectDetector.prototype.train = async function(dataset,epochs, validationSet){
    await this.model.fitDataset(dataset,{epochs : epochs, validationData: validationSet,
        callbacks:
        [  
            tf.callbacks.earlyStopping({monitor: 'val_', minDelta : 0.01, patience: 10, mode: 'max'}) ,
            
            new tf.CustomCallback({
               onEpochEnd: async (epoch, logs) => {
                console.log(tf.memory().numTensors + " tensors")
                console.log(logs)
                console.log(epoch)
                }//,
               // onBatchEnd: ()=> console.log("memory stats : " + tf.memory())
            }),

            
        ]
    })
}

objectDetector.prototype.detect = function(imgBatch, scoreThresh){
    var predictions = this.model.predict(imgBatch)

    return this.yolo.extractBoxesFromPredictedLabelBatch(predictions, scoreThresh)
}

objectDetector.prototype._preprocessXBatch = async function(imgBatch){
    return tf.image.resizeBilinear(imgBatch,[this.inputSize, this.inputSize])
}

objectDetector.prototype.saveModel = async function(dirPath){
    var jsonConfig = JSON.stringify(this.modelConfig);
    var saveResults = await this.model.save("file://" + dirPath)
    fs.writeFileSync(dirPath + "model_cfg.json", jsonConfig);
    return saveResults;
}


objectDetector.prototype.evaluate = async function(batch){
    var predictions = await this.predict(batch)
}


module.exports = objectDetector

