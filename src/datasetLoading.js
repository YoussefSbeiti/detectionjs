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

function splitArrayToBatches(arr, batchSize){
    var splitArray = []
    for(let i = 0; i <arr.length; i += batchSize){
        i+batchSize > arr.length ? splitArray.push(arr.slice(i)) : splitArray.push(arr.slice(i,i+batchSize))
        
    }
    return splitArray
}

shuffle = (array) => array.sort(() => Math.random() - 0.5);


var createLoader = (resize, labelingConfig) => (datasetPath, batchSize) => function*(){

            var parseAnnotationWithLabeler = parseAnnotation( createLabeler(labelingConfig) )
            var loadAndResizeImage = loadImagesAsTensor(resize)
           
            var filePaths = fs.readdirSync(datasetPath).map(fileName => datasetPath + path.sep + fileName)
            var imgPaths = filePaths.flatMap(path => {
                var imgName = /(.+)\.jpg$/g.exec(path)
                return imgName != null ? imgName[1] : [] 
            })
            var batchedImages = splitArrayToBatches(imgPaths, batchSize)
            shuffle(batchedImages)
            for(batch of batchedImages){
                shuffle(batch)
                yield batch.reduce( (dataset, imgName, pathIdx) => {
                            return tf.tidy(()=>{
                                
                                var imgTensor = loadAndResizeImage(fs.readFileSync(imgName +".jpg"));
                                var newImagesBatch = dataset.xs ? dataset.xs.concat(imgTensor) : imgTensor
                                var labelTensor = parseAnnotationWithLabeler(fs.readFileSync( ( imgName + ".txt") , "utf-8"))
                                var newLabelsBatch = dataset.ys ? dataset.ys.concat(labelTensor) : labelTensor
                                
                                dataset.xs ? dataset.ys.dispose() : null //dispose if previous batch
                                dataset.ys ? dataset.xs.dispose() : null //dispose if previous batch
        
                                return {xs: newImagesBatch, ys:newLabelsBatch};
                            })
                        } , {xs: undefined, ys: undefined})
            }
            
                
        
            // var data = filePaths.reduce( (dataset, path, pathIdx) => {
            //     var imgName = /(.+)\.jpg$/g.exec(path)
            //     if(imgName != null){
            //         imgName = imgName[1]
            //         return tf.tidy(()=>{
            //             var imgTensor = loadAndResizeImage(fs.readFileSync(path));
            //             var newImagesBatch = dataset.imagesBatch ? dataset.imagesBatch.concat(imgTensor) : imgTensor
                        
            //             var labelTensor = parseAnnotationWithLabeler(fs.readFileSync( ( imgName + ".txt") , "utf-8"))
            //             var newLabelsBatch = dataset.labelsBatch ? dataset.labelsBatch.concat(labelTensor) : labelTensor
                        
            //             dataset.labelsBatch ? dataset.labelsBatch.dispose() : null
            //             dataset.imagesBatch ? dataset.imagesBatch.dispose() : null

            //             newFilePaths = dataset.filePaths.concat(imgName)
            //             console.log("loaded image number " + pathIdx/2)
            //             return {imagesBatch: newImagesBatch, labelsBatch:newLabelsBatch, filePaths: newFilePaths};
            //         })
            //     }
            //     return dataset; // to handle the case where path is the path of an annotation file
            // } , {imagesBatch: undefined, labelsBatch: undefined, filePaths: []})

            // return {data, filePaths};
}


module.exports = {createLoader, parseAnnotation,loadImagesAsTensor}