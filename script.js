import * as tf from "@tensorflow/tfjs";

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Provide some housing data
const xs = tf.tensor1d([
  7.9, 8.1, 8.3, 8.5, 8.6, 8.4
]);
const ys = tf.tensor1d([
  738967, 742371, 750984, 759598, 763905, 755291
]);

// Train the model using the data provided
model.fit(xs, ys).then(() => {
  const form = document.getElementById("myform");
  const inputText = document.getElementById("inputText");
  const predictPlaceholder = document.getElementById("predict");

  form.addEventListener("submit", e => {
    e.preventDefault();
    // Use the model to predict or to inference
    const output = model.predict(
      tf.tensor2d(
        [parseFloat(inputText.value) / 10], [1, 1]
      ));
    predictPlaceholder.innerHTML = formatting(Array.from(output.dataSync())[0]);
  });
});

const formatting = num => {
  num *= 1000;
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
};
