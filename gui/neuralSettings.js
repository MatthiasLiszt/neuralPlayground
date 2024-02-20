var Settings = {
  layers: 4,
  neurons: [15, 9, 9, 10],
  learningRate: 0.01,
  activation: 'sigmoid',
  f: x => 1/(1 + Math.E**(-x))
};

function loadSettings() {
  var Id = document.getElementById('response');
  Id.textContent = JSON.stringify(Settings);
}