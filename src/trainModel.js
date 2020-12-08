const tf = require('@tensorflow/tfjs-node-gpu')
const path = require("path")
const objectDetector = require('./objectDetector.js')
const {createLoader} = require('./datasetLoading');
const fetch = require('node-fetch')
const fs = require('fs');
const prompt = require('prompt-sync')();
  
(async function(){

    const args = (() => {
        const arguments = {};
        process.argv.slice(2).map( (element) => {
            const matches = element.match( '--([a-zA-Z0-9]+)=(.*)');
            if ( matches ){
                arguments[matches[1]] = matches[2]
                    .replace(/^['"]/, '').replace(/['"]$/, '');
            }
        });
        return arguments;
    })();
    
    //var args = process.argv.slice(2);
    var datasetPath = __dirname + path.sep + ".." + path.sep + "datasets" + path.sep + args.dataset + path.sep 
    var batchSize = parseInt(args.batchSize)
    var epochs = parseInt(args.epochs)

    //var modelConfig  = JSON.parse(fs.readFileSync(cfgPath));
    var classList = fs.readFileSync(datasetPath + "_darknet.labels" , 'utf-8').split("\n")
    var modelConfig = {anchors: [[0.044, 0.052], [0.144, 0.1585] , [0.257, 0.42] , [0.6, 0.27] , [0.751, 0.70]], gridSize:13, numOfClasses: classList.length}

    detector = new objectDetector(modelConfig)
    console.log("Loading weights and Building model. This may take a while...")
    await detector.buildModel("https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/weightsManifest.json" , "https://object-detection-hostedfiles.s3.ca-central-1.amazonaws.com/weights/")
    console.log("model built succesfully")

    console.log("Loading dataset")
    var datasetLoader = createLoader({width: 416, height: 416}, modelConfig)
    var trainSet = tf.data.generator(datasetLoader(datasetPath+ path.sep + "train", batchSize))
    var validSet = tf.data.generator(datasetLoader(datasetPath + path.sep + "valid",1))
    console.log("dataset loaded")

    await detector.train(trainSet, epochs , validSet);

    while(prompt("continue training ?") == "y"){
        await detector.train(trainSet, 5 , validSet);
    }
    
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
    
