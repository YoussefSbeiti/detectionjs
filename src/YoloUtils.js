const tf = global.tf
const width = 13
const num_anchor = 5
const ANCHORS = tf.tensor([0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17] , [5,2]);

grid_y = tf.tidy(() => {

    var linspace = tf.linspace(0, width - 1, width)
    linspace = linspace.reshape([width, 1])

    var grid_y = linspace.tile([1, num_anchor * width])
    return grid_y.reshape([width, width, num_anchor, 1])

})

grid_x = tf.tidy(() => {
    var linspace = tf.linspace(0, width - 1, width)
    linspace = linspace.reshape([width, 1])

    var grid_x = linspace.tile([1, num_anchor])
    grid_x = grid_x.tile([width, 1])
    return grid_x.reshape([width, width, num_anchor, 1])
})


var iou = function(box1s, box2s){
    
    if(box1s.shape[0] == 0 || box2s.shape[0] == 0 ){
        return tf.tensor(0)
    }

    var [box1TopLeft , box1BottomRight] = box1s.split(2,1);
    var [box2TopLeft , box2BottomRight] = box2s.split(2,1);

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
    return ious
}



var extractBoxesFromTrueLabel = function(label){

    var num_class = (label.shape[3] / 5) - 5

    label = label.squeeze().reshape([13, 13, 5, 5 + num_class])

    var objectness_mask = tf.greater(label, tf.zerosLike(label)).any(-1, true).toInt().squeeze();
    objectness_mask = objectness_mask.reshape([13*13*5]).arraySync()

    var boxes = label.slice([0,0,0,0], [-1,-1,-1, 4]);
    var boxsX = boxes.slice([0,0,0,0] , [-1,-1,-1,1]).add(grid_x).div(13)
    var boxsY = boxes.slice([0,0,0,1] , [-1,-1,-1,1]).add(grid_y).div(13)
    var boxsDimensions = boxes.slice([0,0,0,2],[-1,-1,-1,2])
    boxes = tf.concat([boxsX,boxsY,boxsDimensions] , -1).reshape([13*13*5,4])
    boxes = boxes.arraySync()

    var boxes_temp = []
    objectness_mask.forEach((val,i) =>{
        if(val == 1){
            boxes_temp.push(boxes[i])
        }
    })
    boxes_temp = boxes_temp.map(box=>[box[0]-box[2]/2,box[1]-box[3]/2,box[0]+box[2]/2 , box[1]+box[3]/2])
    boxes = tf.tensor(boxes_temp)
    return boxes
}


var extractBoxesFromPredictedLabel = function(outputs){

    outputs = outputs.squeeze()

    const channels = outputs.shape[2]
    const height = outputs.shape[0];
    const width = outputs.shape[1];
    const num_anchor = 5

    const num_class = channels / num_anchor - 5;

    outputs = outputs.reshape([height,width,num_anchor,5+num_class])

    var [boxes, scores, classes] = tf.tidy(() => {

        
        var x = outputs.slice([0, 0, 0, 0], [height, width, 5, 1])
        var y = outputs.slice([0, 0, 0, 1], [height, width, 5, 1])
        var w = outputs.slice([0, 0, 0, 2], [height, width, 5, 1]).flatten()
        var h = outputs.slice([0, 0, 0, 3], [height, width, 5, 1]).flatten()
        var objectness = outputs.slice([0, 0, 0, 4], [height, width, 5, 1])
        var class_probabilities = outputs.slice([0, 0, 0, 5], [height, width, 5, num_class])
        objectness = objectness.sigmoid()
        class_probabilities = class_probabilities.softmax().mul(objectness).reshape([width * width * 5, num_class])

        x = x.sigmoid().add(grid_x).div(tf.scalar(width)).flatten()
        y = y.sigmoid().add(grid_y).div(tf.scalar(height)).flatten()

        w = w.exp().mul(ANCHORS.slice([0, 0], [5, 1]).tile([width * width, 1]).flatten()).div(width)
        h = h.exp().mul(ANCHORS.slice([0, 1], [5, 1]).tile([width * width, 1]).flatten()).div(height)

        var halfWidth = w.div(tf.scalar(2))
        var halfHeight = h.div(tf.scalar(2))

        var boxes = tf.stack([x.sub(halfWidth), y.sub(halfHeight), x.add(halfWidth), y.add(halfHeight)]).transpose()
        var classes = class_probabilities.argMax(1, true);
        var scores = class_probabilities.max(1, true)

        outputs.dispose()

        return [boxes, scores, classes]

    })

    var selected_indices = tf.image.nonMaxSuppression(boxes, scores.squeeze(), 10, 0.5,0.15)
    let selected_boxes = boxes.gather(selected_indices)
    let selected_scores = scores.gather(selected_indices)
    let selected_classes = classes.gather(selected_indices)

    var rslt = [selected_classes, selected_scores, selected_boxes];

    boxes.dispose()
    scores.dispose()
    classes.dispose()
    selected_indices.dispose()
        

    return rslt 
}

var lossFunction =  function(true_y,predicted_y){
    
    var anchors = tf.tensor([0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17] , [5,2]);

    var num_class = (predicted_y.shape[3]/5) - 5

    true_y = true_y.squeeze().reshape([13,13,5,5+ num_class])
    predicted_y = predicted_y.squeeze().reshape([13,13,5,5+num_class])

    var x_true = true_y.slice([0,0,0,0],[13,13,5,1])
    var y_true = true_y.slice([0,0,0,1],[13,13,5,1])
    var w_true = true_y.slice([0,0,0,2],[13,13,5,1])
    var h_true = true_y.slice([0,0,0,3],[13,13,5,1])

        
    var x_predicted = predicted_y.slice([0,0,0,0],[13,13,5,1]).sigmoid()
    var y_predicted = predicted_y.slice([0,0,0,1],[13,13,5,1]).sigmoid()

    var w_predicted = predicted_y.slice([0,0,0,2],[13,13,5,1]).flatten()
    var h_predicted = predicted_y.slice([0,0,0,3],[13,13,5,1]).flatten()

    w_predicted = w_predicted.exp().mul(anchors.slice([0,0] , [5,1]).tile([13*13,1]).flatten()).div(13).reshape([13,13,5,1])
    h_predicted = h_predicted.exp().mul(anchors.slice([0,1] , [5,1]).tile([13*13,1]).flatten()).div(13).reshape([13,13,5,1])

    var objectness_predicted = predicted_y.slice([0,0,0,4],[13,13,5,1]).sigmoid()
    var class_probabilities_predicted = predicted_y.slice([0,0,0,5],[13,13,5,num_class]).softmax()

    var objectness_true = true_y.slice([0,0,0,4],[13,13,5,1])
    var class_probabilities_true = true_y.slice([0,0,0,5],[13,13,5,num_class])

    var objectenss_mask = tf.greater(true_y , tf.zerosLike(true_y)).any(-1,true).toInt()
    var no_objectness_mask = tf.equal(tf.zerosLike(objectenss_mask) , objectenss_mask).toInt()
    
    var term1 = x_predicted.sub(x_true).square().add(y_predicted.sub(y_true).square()).mul(objectenss_mask).sum().mul(tf.scalar(5))
    var term2 = w_predicted.sqrt().sub(w_true.sqrt()).square().add(h_predicted.sqrt().sub(h_true.sqrt()).square()).mul(objectenss_mask).sum().mul(tf.scalar(5))
    var term3 = objectness_predicted.sub(objectness_true).square().mul(objectenss_mask).sum()
    var term4 = objectness_predicted.sub(objectness_true).square().mul(tf.scalar(0.5)).mul(no_objectness_mask).sum()
    var term5 = class_probabilities_predicted.sub(class_probabilities_true).square().mul(objectenss_mask).sum()
    var finalTerm  = term1.add(term2).add(term3).add(term4).add(term5)

    return finalTerm
}

var accuracy = function(true_y , predicted_y){
  
    var [predicted_classes, predicted_scores, predicted_boxes] = extractBoxesFromPredictedLabel(predicted_y)

    var true_boxes = extractBoxesFromTrueLabel(true_y)
   
    var ious = iou(true_boxes, predicted_boxes);

    return ious.mean();
}

module.exports = {loss: lossFunction, accuracy: accuracy}