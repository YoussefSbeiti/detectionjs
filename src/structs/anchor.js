anchor= function (w, h) {
    if (arguments.some(element => element < 0 && element > 1)) return null;
}

bboxIntersection= (anchor) => (bboxDims) => Math.min(anchor[0], bboxDims[0]) * Math.min(anchor[1], bboxDims[1])

bboxUnion= (anchor) => (bboxDims) => bboxDims[0] * bboxDims[1] + anchor[0] * anchor[1] - bboxIntersection(anchor)(bboxDims)

bboxIou= (anchor) => (bboxDims) => bboxIntersection(anchor)(bboxDims) / bboxUnion(anchor)(bboxDims)

bboxRelativeDimensions= (anchor) => (bboxDims) => [bboxDims[0] / anchor[0], bboxDims[1] / anchor[1]]

module.exports = {anchor : anchor, bboxIou : bboxIou, bboxRelativeDimensions : bboxRelativeDimensions}

