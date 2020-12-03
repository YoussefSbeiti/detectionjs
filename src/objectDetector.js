const { data } = require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs-node-gpu')
const fetch = require('node-fetch');
const {createFunctions} = require("./Yolo")
const fs = require("fs")

var objectDetector = function(modelConfig){

    this.modelConfig = modelConfig
    this.yolo = createFunctions(modelConfig.anchors, modelConfig.gridSize, modelConfig.numOfClasses)

}

objectDetector.prototype.buildModel = async function(manifestUrl , weightsUrl){
     
    var numOfClasses = this.modelConfig.numOfClasses
    var numOfAnchors = this.modelConfig.anchors.shape[0]

    function conv2d(kernelSize, filters,strides,name, batchInputShape=undefined, useBias = true, padding = "same"){
        return tf.layers.conv2d({ useBias:useBias, 
                                padding:padding,
                                filters:filters,
                                kernelSize:kernelSize,
                                name:name, 
                                strides:strides,
                                batchInputShape:batchInputShape,
                                kernelRegularizer: tf.regularizers.l2({l2:0.06}),
                                //kernelInitializer: tf.initializers.glorotNormal(),
                                //biasInitializer: tf.initializers.glorotNormal(),
                                weights: [weightsMap[name + "/kernel"] , weightsMap[name + "/bias"] ],
                                trainable:true
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
    model.add(conv2d(3,16,1,"layer1_conv", [1,416,416,3]))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,32,1,"layer2_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,64,1,"layer3_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))

    model.add(conv2d(3,128,1,"layer4_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    
    model.add(conv2d(3,256,1,"layer5_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    
    model.add(conv2d(3,512,1,"layer6_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "same" , poolSize: 2 , strides:1}))
    
    model.add(conv2d(3,1024,1,"layer7_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    
    model.add(conv2d(3,512,1,"layer8_conv"))
    // model.add(tf.layers.batchNormalization())
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    
    //model.add(tf.layers.conv2d({useBias:true, padding:"same",filters:(5+numOfClasses)*numOfAnchors , kernelSize:1,name:"m_outputs0", weights:[finalLayerWeights , finalLayerBiases],strides:1}))
    model.add(conv2d(1,(5+numOfClasses)*numOfAnchors,1,"m_outputs0"))
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: this.yolo.batchLoss,
        metrics : this.yolo.accuracy(this.yolo.extractBoxesFromPredictedLabelBatch, this.yolo.extractBoxesFromTrueLabelBatch)
    })

    console.log("Model built successfully!")
    console.log(model.summary())

    this.model = model

}

/*
*   Fit the model on the samples stored in the classifier's data Object.
*   @param{int} epochs
*   @param{int} batch size
*/
objectDetector.prototype.train = async function(x,y, epochs, batchSize, validationX,validationY){
    await this.model.fit(x, y,{epochs : epochs, batchSize: batchSize ,/*validationData: [validationX, validationY]*/shuffle:true,
        callbacks:
        [  
            //tf.callbacks.earlyStopping({monitor: 'val_', minDelta : 0.009, patience: 10, mode: 'max'}) ,
            
            new tf.CustomCallback({
               onEpochEnd: async (epoch, logs) => {
                //tf.callbacks.earlyStopping({monitor: 'val_acc'})
                console.log(tf.memory().numTensors + " tensors")
                //console.log({'loss' :logs.loss , 'trainLoss' : logs.sequential_2_loss, 'trainAcc' : logs.sequential_2_acc, 'valLoss' : logs.val_sequential_2_loss, 'valAcc' : logs.val_sequential_2_acc})
                console.log(logs)
                console.log(epoch)
                }
            })
        ]
    })
}

objectDetector.prototype.detectBatch = async function(imgBatch){
    //var preprocessedBatch = this._preprocessImage(imgBatch)
    var predictions = await this.model.predict(imgBatch)
    //preprocessedBatch.dispose()
    return this.yolo.extractBoxesFromPredictedLabelBatch(predictions)
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

