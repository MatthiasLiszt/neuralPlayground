// only supports sigmoid function as activation function
function backpropFast4Layer(outputs, Weights, Bias, target, input) {
  var Changed = {w: [] , b: []};

  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // find highest and lowest output
  var indexMaxO = 0;
  //var indexMinO = 0;

  for (var index of outputs.last.keys()) {
    indexMaxO = outputs.last[index] > outputs.last[indexMaxO] ? index : indexMaxO;
  }

  // calculate little deltas
  
  // last layer
  var lastLayerDelta = [];
  for (var i = 0; i < Settings.neurons[3]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (oj - tj) * oj * (1 - oj);

    lastLayerDelta.push(dj);
  }

  // 2nd layer
  var secondLayerSum = 1e-6;
  var maxDelta2ndLayer = 1e-6;
  var indexMaxO2ndLayer = 0;

  for (var index of outputs.last) {
    indexMaxO2ndLayer = outputs.last[index] > indexMaxO2ndLayer ? index : indexMaxO2ndLayer;
  }

  for (var j = 0; j < Settings.neurons[3]; ++j) {
    var w = Weights[2][j][indexMaxO2ndLayer];
    secondLayerSum += w * lastLayerDelta[j];
  }

  var secondLayerDelta = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var sum = secondLayerSum;
    var oi = outputs.O[1][i];
 
    var value = sum * oi * (1 - oi);
    if(!value) console.log(`!!!! secondLayerDelta  !!!! ${value} indexMaxO ${indexMaxO} Weights[2][0].length ${Weights[2][0].length}`);
    secondLayerDelta.push(value);
    maxDelta2ndLayer = maxDelta2ndLayer < value ? value : maxDelta2ndLayer;
  }

  // 1st layer 
  //var guessedDecrease = 0.1;
  var firstLayerDelta = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var guess = maxDelta2ndLayer < 0.5 ? maxDelta2ndLayer ** 2 : 1e-4;
    var oi = outputs.O[0][i];
    
    firstLayerDelta.push(guess * oi * (1 - oi));
  }
  
  // change Weights
  for (var j = 0; j < Settings.neurons[1]; ++j) {
    for (var i = 0; i < Weights[0][j].length; ++i) {
      var ip = input[i];
      Weights[0][j][i] -= firstLayerDelta[j] * ip * Settings.learningRate; 
    }
  }
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    for (var i = 0; i < Weights[1][j].length; ++i) {
      var oi = outputs.O[0][i];
      Weights[1][j][i] -= secondLayerDelta[j] * oi * Settings.learningRate; 
    }
  }

  for (var j = 0; j < Settings.neurons[3]; ++j) {
    for (var i = 0; i < Weights[2][j].length; ++i) {
      var oi = outputs.O[1][i];
      Weights[2][j][i] -= lastLayerDelta[j] * oi * Settings.learningRate; 
    }
  }

  // change Bias
  
  for (var i = 0; i < Bias[0].length; ++i) {
     Bias[0][i] -= firstLayerDelta[i] * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[1].length; ++i) {
    var oi = outputs.O[0][i];
     Bias[1][i] -= secondLayerDelta[i] * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[2].length; ++i) {
     Bias[2][i] -= lastLayerDelta[i] * Settings.learningRate; 
  }
  

  Changed.w = Weights;
  Changed.b = Bias;
  Changed.ld = [firstLayerDelta, secondLayerDelta, lastLayerDelta];
  return Changed;
}