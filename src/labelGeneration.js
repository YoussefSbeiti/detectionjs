const utils = require("./misc")
const Bbox =  require("./structs/bbox")
const Anchor = require("./structs/anchor");

var bestFitAnchorIndex = (anchors) => (bbox) => utils.argMax(anchors.map(anchor => Anchor.bboxIou(anchor)(bbox) ));

var flatLabelIdxGenerator = (numOfAnchors, numOfClasses, gridSize) => (correspondingGridCell, correspondingAnchorIdx) => {
    var gridCellArraySize = numOfAnchors*(5+numOfClasses)
    return correspondingGridCell[0]*gridSize*gridCellArraySize + correspondingGridCell[1]*gridCellArraySize + correspondingAnchorIdx* (5+numOfClasses)
} 

var Labeler = (classList, labelingConfig) => (annotations) => {
    var numOfAnchors = labelingConfig.anchors.length;
    var numOfClasses = classList.length;
    var gridSize = labelingConfig.gridSize;
    var flatLabelLength = gridSize*gridSize*numOfAnchors*(5+numOfClasses)


    var toGridScale = Bbox.transform(coord => coord*gridSize);
    var getBestFitAnchorIdx = bestFitAnchorIndex(labelingConfig.anchors);
    var getFlatLabelIdx = flatLabelIdxGenerator(numOfAnchors, numOfClasses, gridSize)

    var labelArray = Array(flatLabelLength).fill(0)
    annotations.forEach(annotation => {

        var bbox = annotation.bbox;
        var classLabel = annotation.classLabel

        var gridCoords = toGridScale(bbox)
        var correspondingGridCell = Bbox.center(gridCoords).map(coord => Math.floor(coord))
        var inCellCenterCoords = Bbox.center(gridCoords.map(coord => coord - Math.floor(coord) ))
        var correspondingAnchorIdx = getBestFitAnchorIdx(Bbox.dimensions(bbox))
        var correspondingAnchor = labelingConfig.anchors[correspondingAnchorIdx]
        var anchorRelativeDimensions = Anchor.bboxRelativeDimensions(correspondingAnchor)(Bbox.dimensions(bbox))
        var idx = getFlatLabelIdx(correspondingGridCell,correspondingAnchorIdx)
        
        var localLabel = [].concat(inCellCenterCoords,
            anchorRelativeDimensions,
            [1],
            utils.oneHot(classList)(classLabel)
        )
        labelArray = utils.insertAtIndex(labelArray)(idx, localLabel)   
    });

    return labelArray
}


module.exports = { Labeler}