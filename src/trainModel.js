const tf = require('@tensorflow/tfjs-node-gpu')
const path = require("path")
const objectDetector = require('./objectDetector.js')
const {createLoader} = require('./datasetLoading');
const fetch = require('node-fetch')
const fs = require('fs');


// async function loadImagesData(folderUrl, extension){
//     var bboxesRes = await fetch(folderUrl + "boundingBoxes." + extension)
//     var bboxs = await bboxesRes.json()
//     var numberOfSamples = bboxs.length
    

//     var imageSizeRes = await fetch(folderUrl + "imageSize." + extension)
//     var imageSize = await imageSizeRes.json()

//     var imagesRes = await fetch(folderUrl + "images.bin")
//     var imagesBuffer = await imagesRes.arrayBuffer()
//     var imagesTypedArray = new Float32Array(imagesBuffer)
//     var imageSizeInFloats = imageSize*imageSize*3 
//     var images = []
//     for(let i =0; i < numberOfSamples ; i++){
//         var tensor = tf.tensor(imagesTypedArray.slice(i*imageSizeInFloats , imageSizeInFloats*(i+1)))
        

//         tensor = tensor.reshape([imageSize , imageSize , 3])
//         images.push(tensor)
//     }

   
    
//     //var images = await tf.tensor(new Float32Array(imagesBuffer))
//     //images = images.split(numberOfSamples)
//     //images = images.map(img => img.reshape([imageSize,imageSize,3]))
//     console.log("number of images = " + images.length)
//     return [images, bboxs]

//     console.log("loading pictures and labeling them. This may take a while ...")
//     var [images,bboxLbls] = await loadImagesData("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/images/calculator/" , "json")
//     bboxLbls = bboxLbls.map(bboxLbl => Labeler([{bbox: bboxLbl, classIdx: 0}]) ) 
//     detector.data = {x:images, y:bboxLbls}
// }

  
(async function(){
    
    var args = process.argv.slice(2);
    var cfgPath =  __dirname + path.sep + args[0]
    var datasetPath = __dirname + path.sep + args[1] 
    
    var modelConfig  = JSON.parse(fs.readFileSync(cfgPath));
    //var classList = fs.readFileSync(datasetPath + "_darknet.labels" , 'utf-8').split("\n")
    var classList = ["imageTarget"]
    modelConfig = {...modelConfig, numOfClasses: classList.length}

    detector = new objectDetector({...modelConfig, anchors: tf.tensor(modelConfig.anchors) })
    console.log("Loading weights and Building model. This may take a while...")
    await detector.buildModel("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/weightsManifest.json" , "https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/")
    console.log("model built succesfully")

    console.log("Loading dataset")
    var datasetLoader = createLoader({width: 416, height: 416}, modelConfig)
    var dataset = datasetLoader(datasetPath+ path.sep + "train")
    //var validationData = datasetLoader(datasetPath + path.sep + "valid")
    console.log("dataset loaded")

   
    
    await detector.train(dataset.data.imagesBatch, dataset.data.labelsBatch, 40,6 /*validationData.data.imagesBatch, validationData.data.labelsBatch*/);
    
    var modelDirs = fs.readdirSync(datasetPath +  path.sep + "models")
    if(modelDirs.length == 0){
        var modelPath = datasetPath +  path.sep + "models" + path.sep + "iteration1" + path.sep
    }
    else{
        var existingIterations = modelDirs.map(iterationName => parseInt(iterationName.split("iteration")[1]))
        var modelIteration = Math.max(...existingIterations)+1
        var modelPath = datasetPath +  path.sep + "models" + path.sep + "iteration" + modelIteration + path.sep   
    }
    
    var saveResults = await detector.saveModel(modelPath);
})()
    
