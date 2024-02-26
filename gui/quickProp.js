// only supports sigmoid function as activation function
function initGradients4Layer(outputs, Weights, target, input) {

  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // calculate little gradients
  
  // last layer
  var lastLayerDelta = [];
  var lastLayerGradient = [];
  for (var i = 0; i < Settings.neurons[3]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (tj - oj) * oj * (1 - oj);

    lastLayerDelta.push(dj);
    lastLayerGradient.push([]);
    var preliminary = [];
    for(var j = 0; j < Settings.neurons[2]; ++j) {
      var ox = outputs.O[1];
      preliminary.push(dj * ox[j]);
    }
    lastLayerGradient.push(preliminary);
  }

  // 2nd layer 
  var secondLayerDelta = [];
  var secondLayerGradient = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var sum = 0;
    var oi = outputs.O[1][i];
    
    for (var j = 0; j < Settings.neurons[3]; ++j) {
      var w = Weights[2][j][i];
      sum += lastLayerGradient[i][j];
    }
    var dj = (sum * oi * (1 - oi));
    secondLayerDelta.push(dj);
    secondLayerGradient.push([]);
    var preliminary = [];
    for(var j = 0; j < Settings.neurons[1]; ++j) {
      var ox = outputs.O[0];
      preliminary.push(dj * ox[j]);
    }
    secondLayerGradient.push(preliminary);
  }

  // 1st layer 
  var firstLayerDelta = [];
  var firstLayerGradient = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var sum = 0;
    var oi = outputs.O[0][i];
    
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      var w = Weights[1][j][i];
      sum += secondLayerGradient[i][j];
    }
    firstLayerDelta.push(sum * oi * (1 - oi));
    firstLayerGradient.push([]);
    var preliminary = [];
    for(var j = 0; j < Settings.neurons[0]; ++j) {
      var ox = input;
      preliminary.push(dj * ox[j]);
    }
    firstLayerGradient.push(preliminary);
  }

  return {
    gradients: [firstLayerGradient, secondLayerGradient, lastLayerGradient],
    nodeDeltas: [firstLayerDelta, secondLayerDelta, lastLayerDelta], 
  };
}

function initGradients3Layer(outputs, Weights, target, input) {
  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // calculate little gradients
  
  // last layer
  var lastLayerDelta = [];
  var lastLayerGradient = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (tj - oj) * oj * (1 - oj);

    lastLayerDelta.push(dj);
    lastLayerGradient.push([]);
    var preliminary = [];
    for(var j = 0; j < Settings.neurons[1]; ++j) {
      var ox = outputs.O[1];
      preliminary.push(dj * ox[j]);
    }
    lastLayerGradient.push(preliminary);
  }

  // 1st layer 
  var firstLayerDelta = [];
  var firstLayerGradient = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var sum = 0;
    var oi = outputs.O[0][i];
    
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      //var w = Weights[0][j][i];
      sum += lastLayerGradient[i][j];
    }
    firstLayerDelta.push(sum * oi * (1 - oi));
    firstLayerGradient.push([]);
    var preliminary = [];
    for(var j = 0; j < Settings.neurons[0]; ++j) {
      var ox = input;
      preliminary.push(dj * ox[j]);
    }
    firstLayerGradient.push(preliminary);
  }

  return {
    gradients: [firstLayerGradient, lastLayerGradient],
    nodeDeltas: [firstLayerDelta, lastLayerDelta], 
  };
}

// only supports sigmoid function as activation function
function quickprop4Layer(outputs, Weights, Bias, target, input, Gradients, oldGradients, nodeDeltas, oldNodeDeltas) {
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
  newGradients.push([]);
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    var preliminary = [];
    for (var i = 0; i < Weights[1][j].length; ++i) {
      var oi = outputs.O[0][i];
      var newGradient = Gradients[1][j][i] / (oldGradients[1][j][i] - Gradients[1][j][i]);
      preliminary.push(newGradient);
      //Weights[1][j][i] -= secondLayerDelta[j] * oi * Settings.learningRate;
      Weights[1][j][i] -= newGradient * oi * Settings.learningRate;
    }
    newGradients[1].push(preliminary);
  }
  newGradients.push([]);
  for (var j = 0; j < Settings.neurons[3]; ++j) {
    var preliminary = [];
    for (var i = 0; i < Weights[2][j].length; ++i) {
      var oi = outputs.O[1][i];
      var newGradient = Gradients[2][j][i] / (oldGradients[2][j][i] - Gradients[2][j][i]);
      preliminary.push(newGradient);
      //Weights[2][j][i] -= lastLayerDelta[j] * oi * Settings.learningRate;
      Weights[2][j][i] -= newGradient * oi * Settings.learningRate; 
    }
    newGradients[2].push(preliminary);
  }

  // change Bias
  var newNodeDeltas = [];
  newNodeDeltas.push([]);
  for (var i = 0; i < Bias[0].length; ++i) {
    var dj = nodeDeltas[0][i] / (oldNodeDeltas[0][i] - nodeDeltas[0][i]);
    Bias[0][i] -= dj * Settings.learningRate;
    nodeDeltas[0].push(dj); 
  }
  newNodeDeltas.push([]);
  for (var i = 0; i < Bias[1].length; ++i) {
    var dj = nodeDeltas[1][i] / (oldNodeDeltas[1][i] - nodeDeltas[1][i]);
    Bias[1][i] -= dj * Settings.learningRate;
    nodeDeltas[1].push(dj); 
  }
  newNodeDeltas.push([]);
  for (var i = 0; i < Bias[2].length; ++i) {
    var dj = nodeDeltas[2][i] / (oldNodeDeltas[2][i] - nodeDeltas[2][i]);
    Bias[2][i] -= dj * Settings.learningRate;
    nodeDeltas[2].push(dj); 
  }
  

  Changed.w = Weights;
  Changed.b = Bias;
  Changed.gradients = newGradients;
  Changed.nodeDeltas = newNodeDeltas;
  return Changed;
}

// only supports sigmoid function as activation function
function quickprop3Layer(outputs, Weights, Bias, target, input, Gradients, oldGradients, nodeDeltas, oldNodeDeltas) {
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
  newGradients.push([]);
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    var preliminary = [];
    for (var i = 0; i < Weights[1][j].length; ++i) {
      var oi = outputs.O[0][i];
      var newGradient = Gradients[1][j][i] / (oldGradients[1][j][i] - Gradients[1][j][i]);
      preliminary.push(newGradient);
      //Weights[1][j][i] -= secondLayerDelta[j] * oi * Settings.learningRate;
      Weights[1][j][i] -= newGradient * oi * Settings.learningRate;
    }
    newGradients[1].push(preliminary);
  }

  // change Bias
  var newNodeDeltas = [];
  newNodeDeltas.push([]);
  for (var i = 0; i < Bias[0].length; ++i) {
    var dj = nodeDeltas[0][i] / (oldNodeDeltas[0][i] - nodeDeltas[0][i]);
    Bias[0][i] -= dj * Settings.learningRate;
    nodeDeltas[0].push(dj); 
  }
  newNodeDeltas.push([]);
  for (var i = 0; i < Bias[1].length; ++i) {
    var dj = nodeDeltas[1][i] / (oldNodeDeltas[1][i] - nodeDeltas[1][i]);
    Bias[1][i] -= dj * Settings.learningRate;
    nodeDeltas[1].push(dj); 
  }

  Changed.w = Weights;
  Changed.b = Bias;
  Changed.gradients = newGradients;
  Changed.nodeDeltas = newNodeDeltas;
  return Changed;
}

function testQuickprop(steps) {
  var quickprop = Settings.layers == 3 ? quickprop3Layer : quickprop4Layer;
  var backprop = Settings.layers == 3 ? backprop3Layer : backprop4Layer;
  var initGradients = Settings.layers == 3 ? initGradients3Layer : initGradients4Layer;
  var calcAllLayers = Settings.layers == 3 ? calc3Layers : calc4Layers;
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var p5 = patternToInput(DATA[5].pattern);
  p5 = p5.map(x => x = x == 0 ? 0.001 : x);
  var sign = DATA[5].sign;
  var o = calcAllLayers(p5, w, b);
  var res = backprop(o, w, b, sign, p5);
  var beforeLast = initGradients(o, res.w, res.b, sign, p5);
  o = calcAllLayers(p5, w, b);
  res = backprop(o, w, b, sign, p5);
  console.log(JSON.stringify(res.b));
  var last = initGradients(o, res.w, res.b, sign, p5);
  var error = calcError(p5, res.w, res.b, sign, calcAllLayers);
  var best = 1e3;

  for (var i = 0; i < steps; ++i) {
    o = calcAllLayers(p5, w, b);
    //(outputs, Weights, Bias, target, input, Gradients, oldGradients, nodeDeltas, oldNodeDeltas)
    //res = method(o, res.w, res.b, sign, p5);
    res = quickprop(o, res.w, res.b, sign, p5, last.gradients, beforeLast.gradients, last.nodeDeltas, beforeLast.nodeDeltas);
    console.log('q' + i + ' ' + JSON.stringify(res.b));
    beforeLast.gradients = hardcopy(last.gradients);
    beforeLast.nodeDeltas = hardcopy(last.nodeDeltas);
    last.gradients = hardcopy(res.gradients);
    last.nodeDeltas = hardcopy(res.nodeDeltas);
    error = calcError(p5, res.w, res.b, sign, calcAllLayers);
    if (error < best) {
      best = error;
    } 
  }

  function hardcopy(field) {
    var copy = [];
    for (let one of field) {
      copy.push(one);
    }
    return copy;
  }

  error = calcError(p5, res.w, res.b, sign, calcAllLayers);
  return {w: res.w, b: res.b, p: p5, error: error, ld: res.ld, best: best};
}
