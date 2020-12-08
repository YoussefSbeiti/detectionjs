const { tensor } = require("@tensorflow/tfjs-node-gpu");
const tf = require("@tensorflow/tfjs-node-gpu")
const {argMax} = require('./misc.js')

var iou = function(box1s, box2s){
    
    if(box1s.shape[0] == 0 || box2s.shape[0] == 0 ){
        return tf.zeros([box1s.shape[0], 1])
    }

    var [box1TopLeft , box1BottomRight] = box1s.split(2,-1);
    var [box2TopLeft , box2BottomRight] = box2s.split(2,-1);

    var interboxTopLeft = box1TopLeft.maximum(box2TopLeft)
    var interboxBottomRight = box1BottomRight.minimum(box2BottomRight)

    var interboxsWHs = interboxBottomRight.sub(interboxTopLeft).relu()
    //interboxsWHs = interboxsWHs.where(interboxsWHs.greater(tf.zerosLike(interboxsWHs)) , tf.zerosLike(interboxsWHs));
    var interboxArea = interboxsWHs.prod(-1,true)

    var box1Area = box1BottomRight.sub(box1TopLeft).prod(-1,true)
    var box2Area = box2BottomRight.sub(box2TopLeft).prod(-1,true)

    var I = interboxArea
    var U = box1Area.add(box2Area).sub(interboxArea)
    var ious = I.div(U)
    return ious.reshape([box1s.shape[0], box2s.shape[0]])
}


var createFunctions = (anchors, gridSize, numClasses) => {
    if(anchors instanceof Array)
        anchors = tf.tensor(anchors);

    var numAnchors = anchors.shape[0]

    /**
     *  Used for extracting bboxs from a predicted Label.
     */
    var grid_x = tf.tidy(() => {
        var linspace = tf.linspace(0, gridSize - 1, gridSize)
        linspace = linspace.reshape([gridSize, 1])

        var grid_x = linspace.tile([1, gridSize * anchors.shape[0]])
        return grid_x.reshape([gridSize, gridSize, anchors.shape[0], 1])

    })

    var grid_y = tf.tidy(() => {
        var linspace = tf.linspace(0, gridSize - 1, gridSize)
        linspace = linspace.reshape([gridSize, 1])

        var grid_y = linspace.tile([1, anchors.shape[0]])
        grid_y = grid_y.tile([gridSize, 1])
        return grid_y.reshape([gridSize, gridSize, anchors.shape[0], 1])
    })
    /**
     * Splits the given label batch into into tensors  x, y, w, h, objectness, classProbabilities 
     * @param {tf.tensor} label  
     */
    var dissectBatch = function(labelBatch){        
        labelBatch = labelBatch.reshape([-1,gridSize,gridSize, anchors.shape[0], 5 + numClasses])
        var x = labelBatch.slice([0,0, 0, 0, 0], [-1,-1, -1, -1, 1])
        var y = labelBatch.slice([0,0, 0, 0, 1], [-1,-1, -1, -1, 1])
        var w = labelBatch.slice([0,0, 0, 0, 2], [-1,-1,-1, -1, 1])
        var h = labelBatch.slice([0,0, 0, 0, 3], [-1,-1,-1,-1, 1])
        //var angle = labelBatch.slice([0,0, 0, 0, 4], [-1,-1,-1,-1, 1])
        var objectness = labelBatch.slice([0,0, 0, 0, 4], [-1,-1,-1,-1, 1])
        var class_probabilities =  labelBatch.slice([0,0, 0, 0, 5], [-1,-1,-1,-1, numClasses])

        return [x, y, w, h, objectness, class_probabilities]
    }

    /**
     * Splits the batch label tensor, but also does neccesary math operations on each of the extracted values to get value compatible with the ground truth label
     * @param {tf.tensor} labelBatch  
     */
    var extractValuesFromPredictedLabelBatch = function(labelBatch){
        var  [xBatch, yBatch, wBatch, hBatch, objectnessBatch, classProbabilitiesBatch] = dissectBatch(labelBatch)
        xBatch = xBatch.sigmoid()
        yBatch = yBatch.sigmoid()
        wBatch = wBatch.exp().mul(anchors.slice([0,0] , [5,1]))
        hBatch = hBatch.exp().mul(anchors.slice([0,1] , [5,1]))
        objectnessBatch = objectnessBatch.sigmoid()
        classProbabilitiesBatch = classProbabilitiesBatch.softmax()

        return [xBatch, yBatch, wBatch, hBatch, objectnessBatch, classProbabilitiesBatch]        
    }

    var extractValuesFromTrueLabelBatch = function(labelBatch){
        return dissectBatch(labelBatch)
    } 
 
    
    var batchLoss = (true_y,predicted_y) => {

        var batchSize = true_y.shape[0]
        var [x_true, y_true, w_true, h_true, objectness_true, class_probabilities_true] = extractValuesFromTrueLabelBatch(true_y)
        var [x_predicted, y_predicted, w_predicted, h_predicted,objectness_predicted, class_probabilities_predicted] = extractValuesFromPredictedLabelBatch(predicted_y)
        var confidence_predicted = class_probabilities_predicted.mul(objectness_predicted)

        var objectness_mask = tf.greater(objectness_true , 0).toInt()
        var no_objectness_mask = tf.equal(objectness_mask, 0).toInt()
        
        var lambda_coord = 5;
        var lambda_noobj = 0.5;

        //var term1 = x_predicted.sub(x_true).square().add(y_predicted.sub(y_true).square()).mul(objectness_mask).sum().mul(lambda_coord)
        var term1 = tf.squaredDifference(x_predicted, x_true).add(tf.squaredDifference(y_predicted, y_true)).mul(objectness_mask).sum().mul(lambda_coord)
        //var term2 = w_predicted.sqrt().sub(w_true.sqrt()).square().add(h_predicted.sqrt().sub(h_true.sqrt()).square()).mul(objectness_mask).sum().mul(lambda_coord)
        var term2 = tf.squaredDifference(w_predicted.sqrt(), w_true.sqrt()).add(tf.squaredDifference(h_predicted.sqrt(), h_true.sqrt())).mul(objectness_mask).sum().mul(lambda_coord)
        var term3 = tf.squaredDifference(objectness_predicted, objectness_true).mul(objectness_mask).sum()
        var term4 = tf.squaredDifference(objectness_predicted,objectness_true).mul(no_objectness_mask).sum().mul(lambda_noobj)
        var term5 = tf.squaredDifference(confidence_predicted,class_probabilities_true).sum()
        //var term6 = tf.class_probabilities_predicted.sub(class_probabilities_true).square().mul(no_objectness_mask).sum().mul(lambda_noobj)
        var finalTerm  = term1.add(term2).add(term3).add(term4).add(term5).div(batchSize)

        return finalTerm

    }

    var extractBoxesFromPredictedLabelBatch = (labelBatch) => {
        var batchSize = labelBatch.shape[0]
        var splitLabels = labelBatch.split(batchSize, 0)
       
        
        return splitLabels.map( label => {
            var [x, y, w, h, objectness, class_probabilities] = extractValuesFromPredictedLabelBatch(label).map(tn => tn.reshape([gridSize*gridSize*numAnchors, tn.shape[tn.shape.length -1] ]))
            
            x = x.add(grid_x.reshape([gridSize*gridSize*numAnchors,1])).div(gridSize)
            y = y.add(grid_y.reshape([gridSize*gridSize*numAnchors,1])).div(gridSize)
            
            var halfWidth = w.div(tf.scalar(2))
            var halfHeight = h.div(tf.scalar(2))

            var boxes = tf.stack([x.sub(halfWidth), y.sub(halfHeight), x.add(halfWidth), y.add(halfHeight)]).transpose().squeeze()
            var classes = class_probabilities.argMax(-1, true);
            var scores = class_probabilities.mul(objectness).max(-1, true).squeeze();

            var selected_indices = tf.image.nonMaxSuppression(boxes, scores, 20, 0.5,0.4)
            boxes = boxes.gather(selected_indices)
            scores = scores.gather(selected_indices)
            classes = classes.gather(selected_indices)

            return {classes,scores, boxes}
        
        })

    }

    var extractBoxesFromTrueLabelBatch = (labelBatch) => {
        var batchSize = labelBatch.shape[0]
        var splitLabels = labelBatch.split(batchSize, 0)

         
        return splitLabels.map(label => {
            var [x, y, w, h, objectness, class_probabilities] = extractValuesFromTrueLabelBatch(label).map(tn => tn.reshape([-1, tn.shape[tn.shape.length -1]]))
            
            x = x.add(grid_x.reshape([gridSize*gridSize*numAnchors,1])).div(gridSize)
            y = y.add(grid_y.reshape([gridSize*gridSize*numAnchors,1])).div(gridSize)
            
            var objectness_mask = tf.greater(objectness , 0)
            
            var values = [x, y, w, h, objectness_mask, class_probabilities]
            var [x, y, w, h, objectness_mask, class_probabilities] = values.map(tn => tn.arraySync())

            var values = [x, y, w, h, class_probabilities] 
            var [x, y, w, h, class_probabilities] = values.map(arr => arr.filter( (value, idx) => objectness_mask[idx] == 1))

            var boxes = x.map( (x,idx) => [x[0] - w[idx][0]/2 , y[idx][0] - h[idx][0]/2 , x[0] + w[idx][0]/2 , y[idx][0] + h[idx][0]/2])
            boxes = tf.tensor(boxes)
            var classes = class_probabilities.map(oneHot => argMax(oneHot))
            classes = tf.tensor(classes)
            var scores = tf.onesLike(classes)
            
            return {classes,scores, boxes}
        })

    }

    var accuracy = (extractBoxesFromPredictedLabelBatch , extractBoxesFromTrueLabelBatch ) => function(trueBatch , predictedBatch){
        var batchSize = trueBatch.shape[0]

        //var [predictedClassesBatchArray, predictedScoresBatchArray, predictedBoxesBatchArray] = extractBoxesFromPredictedLabelBatch(predictedBatch, scoreThreshold)
        var predictedBatchArray = extractBoxesFromPredictedLabelBatch(predictedBatch)
        
//       var [trueClassesBatchArray, trueScoresBatchArray, trueBoxesBatchArray] = extractBoxesFromTrueLabelBatch(trueBatch)
        var trueBatchArray = extractBoxesFromTrueLabelBatch(trueBatch)

        var total = tf.tensor(0)
        for(let i = 0; i<batchSize;  i++){
            var predictedLabel = predictedBatchArray[i]
            var trueLabel = trueBatchArray[i]

            var predictedBoxes = predictedLabel.boxes
            var trueBoxes = trueLabel.boxes

            var predictedClasses = predictedLabel.classes
            var trueClasses = trueLabel.classes

            var predictedScores = predictedLabel.scores

            var ious = iou(trueBoxes.expandDims(1), predictedBoxes); //expand dims to make it broadcastable
            
            var same_class_mask = trueClasses.expandDims(1).equal(predictedClasses).toInt()

            var labelAcc = ious.mul(same_class_mask).mul(predictedScores).max(-1).mean()
            labelAcc = labelAcc.arraySync() == -Infinity ? tf.tensor(0) : labelAcc
            total = total.add(labelAcc)
        }
        return total.div(batchSize);
    }

    var precision = (extractBoxesFromPredictedLabelBatch , extractBoxesFromTrueLabelBatch ) => function(trueBatch , predictedBatch){
        var batchSize = trueBatch.shape[0]

        //var [predictedClassesBatchArray, predictedScoresBatchArray, predictedBoxesBatchArray] = extractBoxesFromPredictedLabelBatch(predictedBatch, scoreThreshold)
        var predictedBatchArray = extractBoxesFromPredictedLabelBatch(predictedBatch)
        
//       var [trueClassesBatchArray, trueScoresBatchArray, trueBoxesBatchArray] = extractBoxesFromTrueLabelBatch(trueBatch)
        var trueBatchArray = extractBoxesFromTrueLabelBatch(trueBatch)

        var total = tf.tensor(0)
        for(let i = 0; i<batchSize;  i++){
            var predictedLabel = predictedBatchArray[i]
            var trueLabel = trueBatchArray[i]

            var predictedBoxes = predictedLabel.boxes
            var trueBoxes = trueLabel.boxes

            var predictedClasses = predictedLabel.classes
            var trueClasses = trueLabel.classes

            var predictedScores = predictedLabel.scores

            var high_score_boxes = predictedBoxes.gather(predictedScores.greater(0.2).toInt())

            var ious = iou(predictedBoxes.expandDims(1), trueBoxes); //expand dims to make it broadcastable
            //var tp_mask = ious.greater(0.5)
            var same_class_mask = predictedClasses.expandDims(1).equal(trueClasses).toInt()
            ious = ious.mul(same_class_mask).max(-1)
            total = total.add(ious.mul(predictedScores).mean())
            
        }
        

        return total.div(batchSize);
    }
    

    
    return {batchLoss, accuracy, extractBoxesFromPredictedLabelBatch, extractBoxesFromTrueLabelBatch, precision}

}

module.exports = {createFunctions, iou}