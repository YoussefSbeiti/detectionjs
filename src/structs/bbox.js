
var bbox = (arr) =>{
    if(arr.some(element => element < 0 && element > 1) ) return null;
    if(arr[0] + arr[2] > 1 ) arr[2] = 1 - arr[0];
    if(arr[1] + arr[3] > 1) arr[3] = 1-arr[1];
    return arr
}

var center = (bbox) => [bbox[0] , bbox[1]]
var dimensions= (bbox) => [bbox[2],bbox[3]]

var transform = (transform) => (bbox) => bbox.map(transform)

var bboxToString = (bbox) => bbox[0] + " " + bbox[1] + " " + bbox[2] + " " + bbox[3]

module.exports = {bbox: bbox, center : center, dimensions:dimensions, transform: transform, bboxToString}