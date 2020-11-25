const fs = require('fs');
const path = require('path')
//const tf = require('@tensorflow/tfjs-node') 
const {Labeler} = require('./labelGeneration');
const Bbox = require('./structs/bbox')

var Roboflow = {
    darknet: {
        loadDataset : function(datasetPath, resize, labelingConfig){

            var composePath = (base) => (post) => base + path.sep + post 
            var classList = fs.readFileSync(composePath(datasetPath)("_darknet.labels"), 'utf-8').split("\n")
            var createLabel = Labeler(classList, labelingConfig)
            
            var [trainDir, testDir, validateDir] = ["train", "test", "valid"].map( dirType => {
                    var baseDir = composePath(datasetPath)(dirType)
                    return fs.readdirSync(baseDir).map(fileName => composePath(baseDir)(fileName))
            })

 
            var parseAnnotation = function(annotationFileContent){
                if(annotationFileContent.length == "") return tf.zeros([labelingConfig.gridSize, labelingConfig.gridSize, labelingConfig.anchors.length * (5 + classList.length) ]);
                var lines = annotationFileContent.split("\n")
                var _classes = lines.map(line => classList[ line.split(" ")[0] ])
                var bboxs = lines.map( line => Bbox.bbox(line.split(" ").slice(1).map(x => parseFloat(x)) ));
                var annotationsList = _classes.map( (classLbl,i) => { return {classLabel : classLbl, bbox: bboxs[i]} });
                var flatLabel = createLabel(annotationsList)
                var tensor = tf.tensor(flatLabel ,[labelingConfig.gridSize, labelingConfig.gridSize, labelingConfig.anchors.length* (5 + classList.length)]);
                if(tensor.shape.length == 4) {
                    console.log("break")
                }
                return tensor
            }
            
            var [trainData, testData, validateData] = [trainDir, testDir, validateDir].map( dir => {
                var data = {x:[], y:[]}
                dir.forEach((filePath, index) => {
                        var img = /(.+)\.jpg$/g.exec(filePath)
                        if(img != null) {
                            var imgTensor = global.tf.node.decodeJpeg( fs.readFileSync(filePath) )
                            data.x.push( tf.image.resizeBilinear(imgTensor, [resize.height, resize.width ]) );
                            imgTensor.dispose()
                            if(index == 274 * 2) 
                                console.log('break');
                            data.y.push( parseAnnotation( fs.readFileSync( ( img[1] + ".txt") , "utf-8") ) ) ;
                            console.log("annotated image number " + index)
                        }      
                })

                return data;
            });

            return {classList: classList, training: trainData, testing : testData, validation : validateData}
        }
    }
}


module.exports = Roboflow