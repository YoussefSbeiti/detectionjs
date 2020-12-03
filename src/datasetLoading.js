const fs = require('fs');
const path = require('path')
const {createLabeler} = require('./labelGeneration');
const Bbox = require('./structs/bbox')
const tf = require("@tensorflow/tfjs-node-gpu");


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

var loadImagesAsTensor = (resize) => function(imageContent){
    return tf.tidy(() => {
        var imgTensor = tf.node.decodeJpeg( imageContent );
        return tf.image.resizeBilinear(imgTensor, [resize.height, resize.width ]).expandDims() ;
    })
}

var createLoader = (resize, labelingConfig) => (datasetPath) => {
            //var classList = fs.readFileSync(datasetPath + path.sep + "_darknet.labels", 'utf-8').split("\n")
         
            var parseAnnotationWithLabeler = parseAnnotation( createLabeler(labelingConfig) )
            var loadAndResizeImage = loadImagesAsTensor(resize)
           
            var filePaths = fs.readdirSync(datasetPath).map(fileName => datasetPath + path.sep + fileName)

            var data = filePaths.reduce( (dataset, path, pathIdx) => {
                var imgName = /(.+)\.jpg$/g.exec(path)
                if(imgName != null){
                    imgName = imgName[1]
                    return tf.tidy(()=>{
                        var imgTensor = loadAndResizeImage(fs.readFileSync(path));
                        var newImagesBatch = dataset.imagesBatch ? dataset.imagesBatch.concat(imgTensor) : imgTensor
                        
                        var labelTensor = parseAnnotationWithLabeler(fs.readFileSync( ( imgName + ".txt") , "utf-8"))
                        var newLabelsBatch = dataset.labelsBatch ? dataset.labelsBatch.concat(labelTensor) : labelTensor
                        
                        dataset.labelsBatch ? dataset.labelsBatch.dispose() : null
                        dataset.imagesBatch ? dataset.imagesBatch.dispose() : null

                        newFilePaths = dataset.filePaths.concat(imgName)
                        console.log("loaded image number " + pathIdx/2)
                        return {imagesBatch: newImagesBatch, labelsBatch:newLabelsBatch, filePaths: newFilePaths};
                    })
                }
                return dataset; // to handle the case where path is the path of an annotation file
            } , {imagesBatch: undefined, labelsBatch: undefined, filePaths: []})

            return {data, filePaths};
}


module.exports = {createLoader, parseAnnotation}