const tf = require('@tensorflow/tfjs-node-gpu')

//const yoloLabeler = require('./Labeler')
const objectDetector = require('./objectDetector.js')
//const roboflow = require('./utils/Roboflow')
const {loadAndLabelDataset} = require('./datasetLoading');
const {createLabeler} = require('./labelGeneration')
const fetch = require('node-fetch')


async function loadImagesData(folderUrl, extension){
    var bboxesRes = await fetch(folderUrl + "boundingBoxes." + extension)
    var bboxs = await bboxesRes.json()
    var numberOfSamples = bboxs.length
    

    var imageSizeRes = await fetch(folderUrl + "imageSize." + extension)
    var imageSize = await imageSizeRes.json()

    var imagesRes = await fetch(folderUrl + "images.bin")
    var imagesBuffer = await imagesRes.arrayBuffer()
    var imagesTypedArray = new Float32Array(imagesBuffer)
    var imageSizeInFloats = imageSize*imageSize*3 
    var images = []
    for(let i =0; i < numberOfSamples ; i++){
        var tensor = tf.tensor(imagesTypedArray.slice(i*imageSizeInFloats , imageSizeInFloats*(i+1)))
        

        tensor = tensor.reshape([imageSize , imageSize , 3])
        images.push(tensor)
    }

   
    
    //var images = await tf.tensor(new Float32Array(imagesBuffer))
    //images = images.split(numberOfSamples)
    //images = images.map(img => img.reshape([imageSize,imageSize,3]))
    console.log("number of images = " + images.length)
    return [images, bboxs]

    console.log("loading pictures and labeling them. This may take a while ...")
    var [images,bboxLbls] = await loadImagesData("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/images/calculator/" , "json")
    bboxLbls = bboxLbls.map(bboxLbl => Labeler([{bbox: bboxLbl, classIdx: 0}]) ) 
    detector.data = {x:images, y:bboxLbls}
}
  
(async function(){

        
    var labelingConfig = {anchors: [[0.044, 0.052], [0.144, 0.1585] , [0.257, 0.42] , [0.6, 0.27] , [0.751, 0.70]] , gridSize: 13}
    
    var args = process.argv.slice(2);
    var datasetPath = args[0]
    console.log("Loading dataset")
    var dataset = loadAndLabelDataset(datasetPath, {width: 416, height: 416}, labelingConfig)
    console.log("dataset loaded")
    //Labeler = createLabeler({...labelingConfig, numOfClasses: 1})
    detector = new objectDetector({...labelingConfig, anchors: tf.tensor(labelingConfig.anchors), classes : dataset.classList })

    console.log("Loading weights and Building model. This may take a while...")
    await detector.buildModel("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/weightsManifest.json" , "https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/")
    console.log("model built succesfully")
    detector.data= dataset.data
    

    await detector.train(80,6,0.3)
    var saveResults = await detector.model.save("file://" + datasetPath +"/../model")
})()
    
