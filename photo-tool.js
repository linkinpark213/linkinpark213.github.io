"use strict";
    const fs = require("fs");
    const sizeOf = require('image-size');
    const path = "public/images/hello-osaka/thumbnail";
    const outputfile = "public/images/hello-osaka/test-output.json";
    var dimensions;

    fs.readdir(path, function (err, files) {
        if (err) {
            return;
        }
        let arr = [];
        (function iterator(index) {
            if (index == files.length) {
                fs.writeFile(outputfile, JSON.stringify(arr, null, "\t"));
                return;
            }

            fs.stat(path + "/" + files[index], function (err, stats) {
                var filename = files[index];
                
                if (err) {
                    return;
                }
                if (stats.isFile()) {
                    dimensions = sizeOf(path + "/" + files[index]);
                    // arr.push(dimensions.width + 'x' + dimensions.height + ' ' + files[index]);
                    arr.push({
                        width: dimensions.width,
                        height: dimensions.height,
                        filename: files[index],
                        description: ""
                    });
                }
                iterator(index + 1);
            })
        }(0));
    });