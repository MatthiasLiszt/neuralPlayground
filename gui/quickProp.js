// only supports sigmoid function as activation function
function initGradients(outputs, Weights, target, input) {

  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // calculate little gradients
  
  // last layer
  var lastLayerGradient = [];
  for (var i = 0; i < Settings.neurons[3]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (tj - oj) * oj * (1 - oj);

    var preliminary = [];
    for(var j = 0; j < Settings.neurons[2]; ++j) {
      var ox = outputs.O[1];
      preliminary.push(dj * ox[j]);
    }
    lastLayerGradient.push(preliminary);
  }

  // 2nd layer 
  var secondLayerGradient = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var sum = 0;
    var oi = outputs.O[1][i];
    
    for (var j = 0; j < Settings.neurons[3]; ++j) {
      var w = Weights[2][j][i];
      sum += lastLayerGradient[i][j];
    }
    var dj = (sum * oi * (1 - oi));

    var preliminary = [];
    for(var j = 0; j < Settings.neurons[1]; ++j) {
      var ox = outputs.O[0];
      preliminary.push(dj * ox[j]);
    }
    secondLayerGradient.push(preliminary);
  }

  // 1st layer 
  var firstLayerGradient = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var sum = 0;
    var oi = outputs.O[0][i];
    
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      var w = Weights[1][j][i];
      sum += secondLayerGradient[i][j];
    }
    firstLayerDelta.push(sum * oi * (1 - oi));

    var preliminary = [];
    for(var j = 0; j < Settings.neurons[0]; ++j) {
      var ox = input;
      preliminary.push(dj * ox[j]);
    }
    firstLayerGradient.push(preliminary);
  }

  return [firstLayerGradient, secondLayerGradient, lastLayerGradient];
}

// only supports sigmoid function as activation function
function quickprop4Layer(outputs, Weights, Bias, target, input, Gradients, oldGradients) {
  var Changed = {w: [] , b: []};

  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // calculate actual gradients
  
  // change Weights
  var newGradients = [];
  newGradients.push([]);
  for (var j = 0; j < Settings.neurons[1]; ++j) {
    var preliminary = [];
    for (var i = 0; i < Weights[0][j].length; ++i) {
      var ip = input[i];
      var newGradient = Gradients[0][j][i] / (oldGradients[0][j][i] - Gradients[0][j][i]);
      preliminary.push(newGradient);
      Weights[0][j][i] -= newGradient * ip * Settings.learningRate; 
    }
    newGradients[0].push(preliminary);
  }
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    var preliminary = [];
    for (var i = 0; i < Weights[1][j].length; ++i) {
      var oi = outputs.O[0][i];
      var newGradient = Gradients[1][j][i] / (oldGradients[1][j][i] - Gradients[1][j][i]);
      preliminary.push(newGradient);
      //Weights[1][j][i] -= secondLayerDelta[j] * oi * Settings.learningRate;
      Weights[1][j][i] -= newGradient * oi * Settings.learningRate;
    }
  }

  for (var j = 0; j < Settings.neurons[3]; ++j) {
    var preliminary = [];
    for (var i = 0; i < Weights[2][j].length; ++i) {
      var oi = outputs.O[1][i];
      var newGradient = Gradients[2][j][i] / (oldGradients[2][j][i] - Gradients[2][j][i]);
      preliminary.push(newGradient);
      //Weights[2][j][i] -= lastLayerDelta[j] * oi * Settings.learningRate;
      Weights[2][j][i] -= newGradient * oi * Settings.learningRate; 
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
  Changed.gradients = newGradients;
  Changed.ld = [firstLayerDelta, secondLayerDelta, lastLayerDelta];
  return Changed;
}