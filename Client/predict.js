$("#image-selector").change(() => {
    console.log('loading...............');
/*     const reader = new FileReader();
    console.log('reader', reader, event.target)
    reader.onload = () => {
        let dataURL = reader.result;
        console.log('dataurl', dataURL);
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("file")[0];
    reader.readAsDataURL(file); */

    const preview = document.querySelector('img');
    const file = document.querySelector('input[type=file]').files[0];
    console.log('file..........', file)
    const reader = new FileReader();
  
    reader.addEventListener("load", function () {
      // convert image file to base64 string
      preview.src = reader.result;
    }, false);
  
    if (file) {
      reader.readAsDataURL(file);
    }
});

let model;
(async () => {
    console.log('model loading...............');
    model = await tf.loadModel("./tfjs-models/VGG16/model.json");
    console.log('model loaded...............', model);
    $(".progress-bar").hide();
})();

$("#predict-button").click(async () => {
    let image = $("#selected-image").get(0);
    console.log('button...............', image);
    let meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .sub(meanImageNetRGB)
        .reverse(2)
        .expandDims();
    
    // tensor.print()
    console.log('tensor.........', tensor)

    let predictions = await model.predict(tensor).data();
    predictions.print()
    console.log('prediction', predictions);
    
    let top5 = Array.from(predictions)
        .map((p, i) => {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort((a, b) => {
            return b.probability - a.probability;
        }).slice(0, 5);

    $("#prediction-list").empty();
    top5.forEach((p) => {
        $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
    });
});

    /*     let meanImageNetRGB = {
            red: 123.68,
            green: 116.779,
            blue: 103.939
        };
    
        let indices = [
            tf.tensor1d([0], "int32"),
            tf.tensor1d([1], "int32"),
            tf.tensor1d([2], "int32")
        ];
    
        let centeredRGB = {
            red: tf.gather(tensor, indices[0], 2)
                .sub(tf.scalar(meanImageNetRGB.red))
                .reshape([50176]),
            green: tf.gather(tensor, indices[1], 2)
                .sub(tf.scalar(meanImageNetRGB.green))
                .reshape([50176]),
            blue: tf.gather(tensor, indices[2], 2)
                .sub(tf.scalar(meanImageNetRGB.blue))
                .reshape([50176])
        };
    
        let processedTensor = tf.stack([
            centeredRGB.red, centeredRGB.green, centeredRGB.blue
        ], 1)
            .reshape([224, 224, 3])
            .reverse(2)
            .expandDims(); 
        let predictions = await model.predict(processedTensor).data();
            */