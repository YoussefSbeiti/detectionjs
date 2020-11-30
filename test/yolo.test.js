const {createFunctions} = require('../src/Yolo')
const {parseAnnotation} = require('../src/datasetLoading')
const{createLabeler} = require('../src/labelGeneration')
const {tf} = require('./tfUtils')


var anchors = [[0.044, 0.052], [0.144, 0.1585] , [0.257, 0.42] , [0.6, 0.27] , [0.751, 0.70]] 
var gridSize =  13 
var numOfClasses = 1


var labeler = createLabeler({anchors: anchors, numOfClasses: numOfClasses, gridSize: gridSize})
var parseAnnotationWithLabeler = parseAnnotation(labeler)

var yolo = createFunctions(anchors, gridSize, numOfClasses)

//var tn = tf.randomUniform([3 ,gridSize, gridSize, anchors.length*(5+numOfClasses)], 0, 1)
var annotation1 = "0 0.1 0.1 0.2 0.2"
var annotation2 = "0 0.1 0.1 0.23 0.23"
var labelTensor1 = parseAnnotationWithLabeler(annotation1)
var labelTensor2 = parseAnnotationWithLabeler(annotation2)
labelTensor1 = tf.stack([labelTensor1, labelTensor1])
labelTensor2 = tf.stack([labelTensor2, labelTensor2])

var accuracy = yolo.accuracy(yolo.extractBoxesFromTrueLabelBatch, yolo.extractBoxesFromTrueLabelBatch)(labelTensor1, labelTensor2)
console.log(accuracy.arraySync())

test('testing tf util', ()=>{
    expect(accuracy.arraySync()).toBeGreaterThan(0.6)
}) 




// test('', ()=> {
   
//     expect( parseFloat(accuracy.arraySync().toFixed(2) ) ).toBe(0.0);
// })
