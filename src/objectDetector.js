const tf = global.tf
const fetch = require('node-fetch');
const yoloUtils = require("./YoloUtils")

var objectDetector = function(numOfClasses){

    //Sample data to be used for training
    this.data = {x:[], y:[]}
    this.input_size = 416
    this.ANCHORS = tf.tensor([0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17] , [5,2]);
    this.numOfClasses = numOfClasses
    this.numOfAnchors = this.ANCHORS.shape[0]

}

objectDetector.prototype.buildModel = async function(manifestUrl , weightsUrl){

    function conv2d(kernelSize, filters,strides,name, batchInputShape=undefined, useBias = true, padding = "same"){
        return tf.layers.conv2d({ useBias:useBias, 
                                padding:padding,
                                filters:filters,
                                kernelSize:kernelSize,
                                name:name, 
                                strides:strides,
                                batchInputShape:batchInputShape,
                                weights: [weightsMap[name + "/kernel"] , weightsMap[name + "/bias"] ],
                                trainable:true
                            })
    }

    var manifest = await fetch(manifestUrl);
    var weightManifest = await manifest.json();
    var weightsMap = await tf.io.loadWeights(weightManifest, weightsUrl);

    var finalLayerWeights = weightsMap["m_outputs0/kernel"].reshape([512,5,7])
    finalLayerWeights = finalLayerWeights.slice([0,0,0] , [-1,-1,5])
    finalLayerWeights = finalLayerWeights.concat([tf.fill([512,5,this.numOfClasses] , 0)] , -1)
    finalLayerWeights = finalLayerWeights.reshape([1,1,512,this.numOfAnchors*(5+this.numOfClasses)])

    var finalLayerBiases = weightsMap["m_outputs0/bias"].reshape([5,7])
    finalLayerBiases = finalLayerBiases.slice([0,0] , [-1,5])
    finalLayerBiases = finalLayerBiases.concat([tf.fill([5,this.numOfClasses] , 0)] , -1)
    finalLayerBiases = finalLayerBiases.reshape([this.numOfAnchors*(5+this.numOfClasses)])

    var model = tf.sequential()
    model.add(conv2d(3,16,1,"layer1_conv", [1,416,416,3]))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    model.add(conv2d(3,32,1,"layer2_conv"))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    model.add(conv2d(3,64,1,"layer3_conv"))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    model.add(conv2d(3,128,1,"layer4_conv"))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    
    model.add(conv2d(3,256,1,"layer5_conv"))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    model.add(tf.layers.maxPooling2d({ trainable: false,padding: "valid" , poolSize: 2 , strides:2}))
    
    model.add(conv2d(3,512,1,"layer6_conv"))
    model.add(tf.layers.leakyReLU({alpha:0.1}))
    //model.add(tf.layers.maxPooling2d({ trainable: false,padding: "same" , poolSize: 2 , strides:1}))
    
    //model.add(conv2d(3,1024,1,"layer7_conv"))
    //model.add(tf.layers.conv2d({useBias:true, padding:"same",filters:1024, kernelSize:3,name:"layer7_conv", weights: [weightsMap["layer7_conv/kernel"] , weightsMap["layer7_conv/bias"] ],strides:1}))
    //model.add(tf.layers.leakyReLU({alpha:0.1}))
    
    //model.add(conv2d(3,512,1,"layer8_conv"))
    //model.add(tf.layers.conv2d({useBias:true, padding:"same",filters:512, kernelSize:3,name:"layer8_conv", weights: [weightsMap["layer8_conv/kernel"] , weightsMap["layer8_conv/bias"] ],strides:1}))
    //model.add(tf.layers.leakyReLU({alpha:0.1}))
    
    model.add(tf.layers.conv2d({useBias:true, padding:"same",filters:(5+this.numOfClasses)*this.numOfAnchors , kernelSize:1,name:"m_outputs0", weights:[finalLayerWeights , finalLayerBiases], strides:1}))
    //model.add(conv2d(1, (5+this.numOfClasses)*this.numOfAnchors, 1, "m_outputs0"))

    model.compile({
        optimizer: tf.train.adam(0.00005),
        loss: yoloUtils.loss
        //metrics: yoloUtils.accuracy
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
objectDetector.prototype.train = async function(epochs, batchSize, validationSplit){
    var x = tf.stack(this.data.x)
    var y = tf.stack(this.data.y)
    await this.model.fit(x, y ,{epochs : epochs, batchSize: batchSize ,validationSplit: validationSplit,shuffle:true,
        callbacks:
        {
            onEpochEnd: async (epoch, logs) => {
                //tf.callbacks.earlyStopping({monitor: 'val_acc'})
                console.log(tf.memory().numTensors + " tensors")
                //console.log({'loss' :logs.loss , 'trainLoss' : logs.sequential_2_loss, 'trainAcc' : logs.sequential_2_acc, 'valLoss' : logs.val_sequential_2_loss, 'valAcc' : logs.val_sequential_2_acc})
                console.log(logs)
                console.log(epoch)
            },
        }
    })
}

objectDetector.prototype.predict = async function(img){
    return await this._postProcess(await this.model.predict(this._preprocessImage(img)))
}


module.exports = objectDetector
