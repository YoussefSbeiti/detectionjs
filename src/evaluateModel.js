var {bboxToString} = require("./structs/bbox")
const path = require("path")
const fs = require('fs')
const {loadImagesAsTensor} = require("./datasetLoading")
const objectDetector = require("./objectDetector")
const tf = require("@tensorflow/tfjs-node-gpu")
const {PythonShell} = require("python-shell")

var predictionToString = function(prediction){
    var bboxes = prediction.boxes.arraySync()
    var classes = prediction.classes.arraySync()
    var scores = prediction.scores.arraySync()

    var string = ""
    bboxes.forEach((box,i) => {
        string += classes[i] + " " + scores[i] + " " + bboxToString(box) + "\n"    
    });
    return string

}

var loadDetector = async function(modelDirPath){
    var cfgPath = modelDirPath + path.sep + "model_cfg.json"
    var modelConfig = JSON.parse(fs.readFileSync(cfgPath));
    var detector = new objectDetector(modelConfig)
    detector.model = await tf.loadLayersModel("file://" + modelDirPath + path.sep + "model.json")
    return detector
};

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
    var subDirs = ['train', 'test', 'valid']
    var datasetPath = __dirname + path.sep + ".." + path.sep + "datasets" + path.sep + args.dataset
    var modelPath = datasetPath + path.sep + "models" + path.sep + "iteration" + args.iteration 

    var detector = await loadDetector(modelPath)

    //var scoreThreshold = args.scoreThreshold;
    
    subDirs.forEach(dir => {
        var dirPath = datasetPath + path.sep + dir
        var files = fs.readdirSync(dirPath)

        var savePath = modelPath + path.sep + dir + '_predictions'
        try{
            fs.mkdirSync(savePath) // make file where we're going to store predictions
        }catch(e){
        }

        var imgNames = files.flatMap(path_ => {
            var imgName = /(.+)\.jpg$/g.exec(path_)
            return imgName != null ? imgName[1] : [] 
        })

        imgNames.forEach(imgName => {
            var imgPath =  dirPath + path.sep + imgName 
            var imgTensor = loadImagesAsTensor({width:416, height:416})(fs.readFileSync(imgPath + ".jpg"))
            var prediction =  detector.detect(imgTensor)

            var predictionString = predictionToString(prediction[0])
            fs.writeFileSync(savePath + path.sep + imgName+ ".txt", predictionString)
        })
    })
    
})()


