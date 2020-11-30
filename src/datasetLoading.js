const fs = require('fs');
const path = require('path')
const {createLabeler} = require('./labelGeneration');
const Bbox = require('./structs/bbox')
const tf = require("@tensorflow/tfjs-node-gpu")


var parseAnnotation = (labeler) => function(annotationString){
//    if(annotationString.length == "") return tf.zeros([labelingConfig.gridSize, labelingConfig.gridSize, labelingConfig.anchors.length * (5 + classList.length) ]);

    var lines = annotationString.split("\n")
    var classIdxs = lines.map(line => parseInt(line.split(" ")[0]) )
    var bboxs = lines.map( line => Bbox.bbox(line.split(" ").slice(1).map(x => parseFloat(x)) ));
    var annotationsList = classIdxs.map( (classIdx,i) => { return {classIdx : classIdx, bbox: bboxs[i]} });
    var labelTensor = labeler(annotationsList)
    //var tensor = tf.tensor(flatLabel ,[labelingConfig.gridSize, labelingConfig.gridSize, labelingConfig.anchors.length* (5 + classList.length)]);
    
    return labelTensor
}

function loadAndLabelDataset(datasetPath, resize, labelingConfig){
            var classList = fs.readFileSync(datasetPath + path.sep + "_darknet.labels", 'utf-8').split("\n")
         
            var parseAnnotationWithLabeler = parseAnnotation( createLabeler({...labelingConfig, numOfClasses: classList.length}) )
           
            var filePaths = fs.readdirSync(datasetPath).map(fileName => datasetPath + path.sep + fileName)

            var data = {x:[], y:[]}
            filePaths.forEach((filePath, index) => {
                    var img = /(.+)\.jpg$/g.exec(filePath)
                    if(img != null) {
                        var imgTensor = tf.node.decodeJpeg( fs.readFileSync(filePath) )
                        data.x.push( tf.image.resizeBilinear(imgTensor, [resize.height, resize.width ]) );
                        imgTensor.dispose()
                        
                        var labelTensor = parseAnnotationWithLabeler(fs.readFileSync( ( img[1] + ".txt") , "utf-8"))
                        //var labelTensor = tf.tensor(flatLabel ,[labelingConfig.gridSize, labelingConfig.gridSize, labelingConfig.anchors.length* (5 + classList.length)]);
                        data.y.push( labelTensor ) ;
                        console.log("annotated image number " + index)
                    }      
            })

            return { data, classList};
}


module.exports = {loadAndLabelDataset, parseAnnotation}