const tf = require('@tensorflow/tfjs-node-gpu')

const utils = require("./misc")
const Bbox =  require("./structs/bbox")
const Anchor = require("./structs/anchor");

var bestFitAnchorIndex = (anchors) => (bbox) => utils.argMax(anchors.map(anchor => Anchor.bboxIou(anchor)(bbox) ));

var flatLabelIdxGenerator = (numOfAnchors, numOfClasses, gridSize) => (correspondingGridCell, correspondingAnchorIdx) => {
    var gridCellArraySize = numOfAnchors*(5+numOfClasses)
    return correspondingGridCell[0]*gridSize*gridCellArraySize + correspondingGridCell[1]*gridCellArraySize + correspondingAnchorIdx* (5+numOfClasses)
} 

var createLabeler = (labelingConfig) => (annotations) => {
    var numOfAnchors = labelingConfig.anchors.length;
    var numOfClasses = labelingConfig.numOfClasses;
    var gridSize = labelingConfig.gridSize;
    var flatLabelLength = gridSize*gridSize*numOfAnchors*(5+numOfClasses)

    var toGridScale = Bbox.transform(coord => coord*gridSize);
    var getBestFitAnchorIdx = bestFitAnchorIndex(labelingConfig.anchors);
    var getFlatLabelIdx = flatLabelIdxGenerator(numOfAnchors, numOfClasses, gridSize)

    var labelArray = Array(flatLabelLength).fill(0)

    annotations.forEach(annotation => {

        var bbox = annotation.bbox;
        var classIdx = annotation.classIdx

        if(isNaN(classIdx)) return tf.zeros([gridSize, gridSize, numOfAnchors * (5 + numOfClasses) ]);

        var gridCoords = toGridScale(bbox)
        var correspondingGridCell = Bbox.center(gridCoords).map(coord => Math.floor(coord))
        var inCellCenterCoords = Bbox.center(gridCoords.map(coord => coord - Math.floor(coord) ))
        var correspondingAnchorIdx = getBestFitAnchorIdx(Bbox.dimensions(bbox))
        var correspondingAnchor = labelingConfig.anchors[correspondingAnchorIdx]
        //var anchorRelativeDimensions = Anchor.bboxRelativeDimensions(correspondingAnchor)(Bbox.dimensions(bbox))
        var idx = getFlatLabelIdx(correspondingGridCell,correspondingAnchorIdx)
        
        var localLabel = [].concat(Bbox.center(bbox),
            Bbox.dimensions(bbox),
            [1],
            utils.oneHot(numOfClasses)(classIdx)
        )
        //labelArray = labelArray.concat([localLabel, correspondingGridCell.concat(correspondingAnchorIdx)])
        
        labelArray = utils.insertAtIndex(labelArray)(idx, localLabel)   
    });

    return tf.tensor(labelArray, [gridSize, gridSize, numOfAnchors* (5 + numOfClasses)])
}


module.exports = { createLabeler}