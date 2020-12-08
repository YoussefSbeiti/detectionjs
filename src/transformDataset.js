const fs = require('fs')
const fse = require('fs-extra');
const path = require('path');

(async function(){
    
    var args = process.argv.slice(2);
    var datasetPath = __dirname + path.sep + args[0] 
    var classMappingObj =  JSON.parse(fs.readFileSync(__dirname + path.sep + args[1]));
    var newDirName = args[2]

    var oldClassList =  fs.readFileSync(datasetPath + "_darknet.labels" , 'utf-8').split("\n")
    var newClassList = classMappingObj.newClassList
    var mappings = classMappingObj.mappings

    var newDirPath = datasetPath+ path.sep + ".." + path.sep + newDirName
    fs.mkdirSync(newDirPath);

    fs.writeFileSync(newDirPath + path.sep + "_darknet.labels", newClassList.join("\n"));

    ['train', 'valid', 'test'].map(dir => {
       fse.copySync(datasetPath + path.sep + dir, newDirPath + path.sep + dir)

        var files = fs.readdirSync(newDirPath + path.sep + dir)
        var annotationFiles = files.filter(fileName => fileName.includes(".txt"))
        
        annotationFiles.forEach(annotationFile => {
            var filePath = newDirPath + path.sep + dir + path.sep + annotationFile
            var data = fs.readFileSync(filePath, 'utf-8');
            if(! annotationFile == ""){

                var annotations = data.split("\n")
                var newAnnotations = annotations.map(annotation => {
                    var currClassIdx = annotation.split(" ")[0]
                    var newClassIdx = newClassList.indexOf(mappings[oldClassList[currClassIdx]])
                    return annotation.replace(new RegExp("^" + currClassIdx), newClassIdx)
                })
                var newAnnotationsString = newAnnotations.join("\n")
                
                if(newAnnotationsString != "-1")
                    fs.writeFileSync(filePath, newAnnotationsString, 'utf-8');
            }
        })
    });
})()
    