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

  // calculate little deltas
  
  // last layer
  var randomNeuron = Math.floor(Math.random() * outputs.last.length);
  var oj = outputs.last[randomNeuron];
  var tj = targets[randomNeuron];
  var lastLayerDelta = (oj - tj) * oj * (1 - oj);

  // 2nd layer
  randomNeuron = Math.floor(Math.random() * outputs.O[1].length);
  var SecondLayerSum = 0;
  for (var j = 0; j < Settings.neurons[3]; ++j) {
    var w = Weights[2][j][randomNeuron];
    SecondLayerSum += w * lastLayerDelta;
  }

  var oi = outputs.O[1][randomNeuron];
  var secondLayerDelta = SecondLayerSum * oi * (1 - oi);

  // 1st layer 
  randomNeuron = Math.floor(Math.random() * outputs.O[0].length);
  var FirstLayerSum = 0;
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    var w = Weights[1][j][randomNeuron];
    FirstLayerSum += w * lastLayerDelta;
  }

  var oi = outputs.O[1][randomNeuron];
  var firstLayerDelta = FirstLayerSum * oi * (1 - oi);
  
  // change Weights
  for (var j = 0; j < Settings.neurons[1]; ++j) {
    for (var i = 0; i < Weights[0][j].length; ++i) {
      var ip = input[i];
      Weights[0][j][i] -= firstLayerDelta * ip * Settings.learningRate; 
    }
  }
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    for (var i = 0; i < Weights[1][j].length; ++i) {
      var oi = outputs.O[0][i];
      Weights[1][j][i] -= secondLayerDelta * oi * Settings.learningRate; 
    }
  }

  for (var j = 0; j < Settings.neurons[3]; ++j) {
    for (var i = 0; i < Weights[2][j].length; ++i) {
      var oi = outputs.O[1][i];
      Weights[2][j][i] -= lastLayerDelta * oi * Settings.learningRate; 
    }
  }

  // change Bias
  
  for (var i = 0; i < Bias[0].length; ++i) {
     Bias[0][i] -= firstLayerDelta * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[1].length; ++i) {
    var oi = outputs.O[0][i];
     Bias[1][i] -= secondLayerDelta * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[2].length; ++i) {
     Bias[2][i] -= lastLayerDelta * Settings.learningRate; 
  }
  
  Changed.w = Weights;
  Changed.b = Bias;
  Changed.ld = [firstLayerDelta, secondLayerDelta, lastLayerDelta];
  return Changed;
}

function backpropFast3Layer(outputs, Weights, Bias, target, input) {
  var Changed = {w: [] , b: []};
  
  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // calculate little deltas
  
  // last layer
  var randomNeuron = Math.floor(Math.random() * outputs.last.length);
  var oj = outputs.last[randomNeuron];
  var tj = targets[randomNeuron];
  var lastLayerDelta = (oj - tj) * oj * (1 - oj);

  // 1st layer 
  randomNeuron = Math.floor(Math.random() * outputs.O[0].length);
  var FirstLayerSum = 0;
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    var w = Weights[1][j][randomNeuron];
    FirstLayerSum += w * lastLayerDelta;
  }

  var oi = outputs.O[0][randomNeuron];
  var firstLayerDelta = FirstLayerSum * oi * (1 - oi);
  
  // change Weights
  for (var j = 0; j < Settings.neurons[1]; ++j) {
    for (var i = 0; i < Weights[0][j].length; ++i) {
      var ip = input[i];
      Weights[0][j][i] -= firstLayerDelta * ip * Settings.learningRate; 
    }
  }

  for (var j = 0; j < Settings.neurons[3]; ++j) {
    for (var i = 0; i < Weights[2][j].length; ++i) {
      var oi = outputs.O[1][i];
      Weights[2][j][i] -= lastLayerDelta * oi * Settings.learningRate; 
    }
  }

  // change Bias
  
  for (var i = 0; i < Bias[0].length; ++i) {
     Bias[0][i] -= firstLayerDelta * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[1].length; ++i) {
     Bias[1][i] -= lastLayerDelta * Settings.learningRate; 
  }
  
  Changed.w = Weights;
  Changed.b = Bias;
  Changed.ld = [firstLayerDelta, lastLayerDelta];
  return Changed;
}
