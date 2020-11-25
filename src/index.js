global.tf = require('@tensorflow/tfjs-node-gpu')

//const yoloLabeler = require('./Labeler')
const objectDetector = require('./objectDetector.js')
//const roboflow = require('./utils/Roboflow')
const Roboflow = require('./datasetLoading');
  
(async function(){

        
    var labelingConfig = {anchors: [[0.044, 0.052], [0.144, 0.1585] , [0.257, 0.42] , [0.6, 0.27] , [0.751, 0.70]] , gridSize: 13}
    
    var args = process.argv.slice(2);
    console.log("Loading dataset")
    //await global.tf.setBackend('cpu')
    var dataset = Roboflow.darknet.loadDataset(args[0], {width: 416, height: 416}, labelingConfig)
    console.log("dataset loaded")

    // console.log("LabelingData")
    // var labelingConfig = {classList: dataset.classList, anchors: [[0.044, 0.052], [0.144, 0.1585] , [0.257, 0.42] , [0.6, 0.27] , [0.751, 0.70]] , gridSize: 13}

    // var trainingData = dataset.training.y.map(createTensorLabel(createLabel))
    // console.log(trainingData)


    // // var name = args[0]
    detector = new objectDetector(dataset.classList.length)

    console.log("Loading weights and Building model. This may take a while...")
    await detector.buildModel("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/weightsManifest.json" , "https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/")
    console.log("model built succesfully")

    // console.log("loading pictures and labeling them. This may take a while ...")
    // var [images,bboxLbls] = await loadImagesData("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/images/" + name + "/" , "json")
    detector.data= dataset.training
    await detector.train(15,10,0.3)
    // var saveResults = await detector.model.save("file://../models/" + name + "-model/")
})()
    
