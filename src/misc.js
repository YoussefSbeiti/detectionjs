var elementWiseSubtract = (arr1) => (arr2) => arr1.map( (element, idx) => element - arr2[idx] )
//Array.prototype.subtract = (arr2) => this.map( (element, idx) => element - arr2[idx] )

var argMax = (array) => array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
var oneHot = (classList) => (classLbl) => Array(classList.indexOf(classLbl)).fill(0).concat( [1], Array(classList.length -(classList.indexOf(classLbl) + 1)).fill(0) ) 

var insertAtIndex = (array) => (idx, subArray) => {
    //if(subArray.length != range[1]) return array;
    return [].concat(array.slice(0,idx),
            subArray,
            array.slice(idx + subArray.length)
        )
}

function arrayEquals(a, b) {
    return Array.isArray(a) &&
      Array.isArray(b) &&
      a.length === b.length &&
      a.every((val, index) => val === b[index]);
}

module.exports = {elementWiseSubtract, argMax, oneHot, insertAtIndex, insertAtIndex, arrayEquals}