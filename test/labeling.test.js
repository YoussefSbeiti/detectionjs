const {Labeler} = require('../src/labelGeneration');

labelingConfig = {
    anchors: [[0.5,1] , [1,0.5], [0.5,0.5], [0.5,0.6] , [0.1,0.4]],
    gridSize : 11
}
classList =  ['1', '2', '3' , '4', '5', '6', '7']
var annotations= [{bbox: [0,0, 0.1,0.1], classLabel : "5"},  {bbox: [0.77,0.9, 0.1,0.4], classLabel : "1"}]
var outputLengthCheck = (length)=>{
    return labelingConfig.gridSize*labelingConfig.gridSize*labelingConfig.anchors.length*(5+classList.length) == length 
}
var outputTypeCheck = (output) => output.every(element => typeof element === 'number')

test('test', () => {
    var createLabel = Labeler(classList, labelingConfig)
    var label = createLabel(annotations) 
    console.log(label.slice(7250))
    console.log(outputLengthCheck(label.length))
    console.log(outputTypeCheck(label))
    expect( outputTypeCheck(label) && outputLengthCheck(label.length) )
    });

