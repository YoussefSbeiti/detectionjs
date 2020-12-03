var {bboxtoString} = require("./structs/bbox")
const path = require("path")
const fs = require('fs')
const objectDetector = require("./objectDetector")

var predictionToString = function(prediction){
    var boxes = annotation.boxes.arraySync()
    var classes = annotation.classes.arraySync()
    var scores = annotation.scores.arraySync()

    var string = ""
    boxes.forEach((box,i) => {
        string += classes[i] + " " + scores + " " + bboxtoString(box) + "\n"    
    });

}

var evaluateDetector = async function(detector, datasetPath, datasetLoader){
        var dataset = datasetLoader(datasetPath)
        var predictionsBatchArray = await detector.detectBatch(dataset.imagesBatch)
        predictionsBatchArray.forEach((prediction, idx) => {
            predictionString = predictionToString(prediction)
            fs.writeFile(datasetPath + path.sep + dataset.filePaths[idx] + ".txt", predictionString)
        })
}

var loadDetector = async function(modelDirPath){
    var cfgPath = modelDirPath + "model_cfg.json"
    var modelConfig = JSON.parse(fs.readFileSync(cfgPath));
    var detector = new objectDetector(modelConfig)
}

(async function(){
    var args = process.argv.slice(2);
    var validationOrTest = args[0]
    var datasetPath = __dirname + path.sep + args[1] 
})()