// only supports sigmoid function as activation function
function backprop4Layer(outputs, Weights, Bias, target, input) {
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
  var lastLayerDelta = [];
  for (var i = 0; i < Settings.neurons[3]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (oj - tj) * oj * (1 - oj);

    lastLayerDelta.push(dj);
  }

  // 2nd layer 
  var secondLayerDelta = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var sum = 0;
    var oi = outputs.O[1][i];
    
    for (var j = 0; j < Settings.neurons[3]; ++j) {
      var w = Weights[2][j][i];
      sum += w * lastLayerDelta[j];
    }
    secondLayerDelta.push(sum * oi * (1 - oi));
  }

  // 1st layer 
  var firstLayerDelta = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var sum = 0;
    var oi = outputs.O[0][i];
    
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      var w = Weights[1][j][i];
      sum += w * secondLayerDelta[j];
    }
    firstLayerDelta.push(sum * oi * (1 - oi));
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

// only supports sigmoid function as activation function
function backprop3Layer(outputs, Weights, Bias, target, input) {
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
  var lastLayerDelta = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (oj - tj) * oj * (1 - oj);

    lastLayerDelta.push(dj);
  }

  // 1st layer 
  var firstLayerDelta = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var sum = 0;
    var oi = outputs.O[0][i];
    
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      var w = Weights[1][j][i];
      sum += w * lastLayerDelta[j];
    }
    var ch = sum * oi * (1 - oi);
    if (isNaN(ch)) {console.log('first layer ' + i + ' ' + j)}
    firstLayerDelta.push(ch);
  }

  // change Weights
  for (var j = 0; j < Settings.neurons[1]; ++j) {
    for (var i = 0; i < Weights[0][j].length; ++i) {
      var oi = input[i];
      Weights[0][j][i] -= firstLayerDelta[j] * oi * Settings.learningRate; 
    }
  }
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    for (var i = 0; i < Weights[1][j].length; ++i) {
      var oi = outputs.O[0][i];
      Weights[1][j][i] -= lastLayerDelta[j] * oi * Settings.learningRate; 
    }
  }

  // change Bias
  
  for (var i = 0; i < Bias[0].length; ++i) {
     Bias[0][i] -= firstLayerDelta[i] * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[1].length; ++i) {
     Bias[1][i] -= lastLayerDelta[i] * Settings.learningRate; 
  }
  

  Changed.w = Weights;
  Changed.b = Bias;
  Changed.ld = [firstLayerDelta, lastLayerDelta];
  return Changed;
}

// only supports sigmoid function as activation function
function initGradients3Layer(outputs, target, input, Weights) {
  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.01;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // calculate little gradients
  var Init = [[],[]];
  // last layer
  var lastLayerDelta = [];
  var lastLayerGradient = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (oj - tj) * oj * (1 - oj);

    lastLayerDelta.push(dj);
    if(isNaN(dj)){console.log(`NaN lastlayer ${i}`)}
    var init = [];
    var preliminary = [];
    for(var j = 0; j < outputs.O[0].length; ++j) {
      var ox = outputs.O[0];
      preliminary.push(dj * ox[j]);
      init.push(0);
    }
    lastLayerGradient.push(preliminary);
    Init[1].push(init);
  }

  // 1st layer
  var firstLayerDelta = [];
  var firstLayerGradient = [];
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    var sum = 0;
    var oi = outputs.O[0][i];
    
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      var w = Weights[1][j][i];
      sum += lastLayerDelta[j] * w;
    }
    var dj = sum * oi * (1 - oi);
    firstLayerDelta.push(dj);
    if(isNaN(dj)){console.log(`NaN firstlayer ${i}`)}
    var init = [];
    var preliminary = [];
    for(var j = 0; j < Settings.neurons[0]; ++j) {
      var ox = input;
      preliminary.push(dj * ox[j]);
      init.push(0);
    }
    firstLayerGradient.push(preliminary);
    Init[0].push(init);
  }

  return {
    gradients: [firstLayerGradient, lastLayerGradient],
    nodeDeltas: [firstLayerDelta, lastLayerDelta],
    init: [Init[0], Init[1]]
  };
}


function updateByGradients3Layer(Weights, Bias, NodeDeltas, Gradients, BatchLength){
  var Changed = {};
  // change Weights
  for (var j = 0; j < Settings.neurons[1]; ++j) {
    for (var i = 0; i < Weights[0][j].length; ++i) {
      Weights[0][j][i] -= (Gradients[0][j][i] / BatchLength) * Settings.learningRate; 
    }
  }
  for (var j = 0; j < Settings.neurons[2]; ++j) {
    for (var i = 0; i < Weights[1][j].length; ++i) {
      Weights[1][j][i] -= (Gradients[1][j][i] / BatchLength) * Settings.learningRate; 
    }
  }

  // change Bias
  for (var i = 0; i < Bias[0].length; ++i) {
    Bias[0][i] -= (NodeDeltas[0][i] / BatchLength) * Settings.learningRate; 
  }
  for (var i = 0; i < Bias[1].length; ++i) {
    Bias[1][i] -= (NodeDeltas[1][i] / BatchLength) * Settings.learningRate; 
  }

  Changed.w = Weights;
  Changed.b = Bias;
  Changed.gradients = Gradients;
  Changed.nodeDeltas = NodeDeltas;
  return Changed;
}

function learnOne(steps) {
  var w = initWeights(0.05);
  var b = initBias(0.05);
  var p5 = patternToInput(DATA[5].pattern);
  var o = calc4Layers(p5, w, b);
  console.log(JSON.stringify(o));
  var res = backprop4Layer(o, w, b, DATA[5].sign, p5);
  // console.log(JSON.stringify(res.w));
  for (var i = 0; i < steps; ++i) {
    o = calc4Layers(p5, res.w, res.b);
    res = backprop4Layer(o, res.w, res.b, DATA[5].sign, p5);
  }
  dumpWeights(res.w);
  var e = calcError(p5, res.w, res.b, DATA[5].sign);
  console.log(`final error ${e} after ${steps} steps`)
  return {w: res.w, b: res.b, p: p5, ld: res.ld};
}

function testBack() {
  var w = initWeights(0.1);
  var b = initBias(0.1);
  var p5 = patternToInput(DATA[5].pattern);
  p5 = p5.map(x => x = x == 0 ? 0.001 : x);
  var o = calc4Layers(p5, w, b);
  var res = backprop4Layer(o, w, b, DATA[5].sign, p5);
  // dumpWeights(res.w);
  return {w: res.w, b: res.b, p: p5};
}

function testItBack() {
  // reduce DATA to learn
  //var patch = DATA.slice(DATA.length - 4);
  var patch = DATA;
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var sample = DATA[2];
  var px = patternToInput(sample.pattern);
  
  var o = calc4Layers(px, w, b);
  var res = backprop4Layer(o, w, b, sample.sign, px);

  // accuracy parameter seems to be critical for the result -- so far 0.07 does well
  res = trainPatchRandomly(res, patch, 0.075);
  var rounds = res.rounds;

  var right = checkLearningResults(res, patch);

  return {w: res.w, b: res.b, p: px, right: right/patch.length, rounds: rounds, patch: patch};
}

function debugMiniBack() {
  var patch = DATA;
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var sample = DATA[2];
  var px = patternToInput(sample.pattern);
  
  var o = calc3Layers(px, w, b);
  var hardcopy = JSON.parse(JSON.stringify(o));
  var weights = JSON.parse(JSON.stringify(w));
  var res = backprop3Layer(o, w, b, sample.sign, px);
  
  console.log(JSON.stringify(res.ld));

  var debug = initGradients3Layer(hardcopy, sample.sign, px, weights);
  console.log(JSON.stringify(debug.nodeDeltas));
}

function testMiniBack() {
  var patch = DATA;
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var sample = DATA[2];
  var px = patternToInput(sample.pattern);
  
  var o = calc3Layers(px, w, b);
  var res = backprop3Layer(o, w, b, sample.sign, px);

  // accuracy parameter seems to be critical for the result -- so far 0.07 does well
  res = trainPatchUsingMinibatches(res, patch, 0.09, 0.1, 3e4);
  var rounds = res.rounds;

  var right = checkLearningResults(res, patch);

  return {w: res.w, b: res.b, p: px, right: right/patch.length, rounds: rounds, patch: patch};
}

function trainPatch(WeightsBias, patch, accuracy) {
  var res = WeightsBias;
  var rounds = 0;
  var success = false;
  for (var k = 0; k < 1024; ++k) {
    for (var one of patch) {
      var p = patternToInput(one.pattern);
      while(calcError(p, res.w, res.b, one.sign) > accuracy) {
        o = calc4Layers(p, res.w, res.b);
        res = backprop4Layer(o, res.w, res.b, one.sign, p);
        ++rounds;
        if(!(rounds%2.5e4)) success = true;
      }
      if(success) console.log(`learning round ${rounds}`);
      success = false;
    }
  }
  return {w: res.w, b: res.b, rounds: rounds}
}

function trainPatchRandomly(WeightsBias, patch, accuracy) {
  var res = WeightsBias;
  var rounds = 0;
  var success = false;
  for (var k = 0; k < 2048 * patch.length; ++k) {
    var x = Math.floor(Math.random() * (patch.length + 1)) % patch.length;
    var one = patch[x];
    var p = patternToInput(one.pattern);
    while(calcError(p, res.w, res.b, one.sign) > accuracy) {
      o = calc4Layers(p, res.w, res.b);
      res = backprop4Layer(o, res.w, res.b, one.sign, p);
      ++rounds;
      if(!(rounds%2.5e4)) success = true;
    }
    if(success) console.log(`learning round ${rounds}`);
    success = false;
    
  }
  return {w: res.w, b: res.b, rounds: rounds}
}

function learnNewDigit(WeightsBias, digit, patch, accuracy) {
  var newPatch = patch;
  var i = 0;
  while(DATA[i].sign != digit){
    ++i;
  }

  newPatch.push(DATA[i]);
  var res = trainPatch(WeightsBias, newPatch, accuracy);
  var rounds = res.rounds;
  var right = checkLearningResults(res, newPatch);

  console.log(JSON.stringify(newPatch));
  return {w: res.w, b: res.b, rounds: rounds, right: right}
}

function trainPatchUsingMinibatches(WeightsBias, patch, accuracy, batchSize, maxUnit) {
  var res = WeightsBias;
  var rounds = 0;
  var success = false;
  var k = 0;

  // create mini batches
  var batches = [];
  var numberOfBatches = Math.floor(1/batchSize);
  for (var i = 0; i < numberOfBatches; ++i) {
    batches.push([]);
    for (var j = 0; j < patch.length * batchSize; ++j) {
      var x = Math.floor(Math.random() * (patch.length + 1)) % patch.length;
      var one = patch[x];
      batches[i].push({sign: one.sign, input: patternToInput(one.pattern)});
    }
  }
  console.log(JSON.stringify(batches[0]));

  while (k < 128) {
    var y = Math.floor(Math.random() * (batches.length + 1)) % batches.length;
    var innerRounds = 0;
    var error = accuracy * 2;
    
    while(error > accuracy && innerRounds < maxUnit) {
      var error = 0;
      var px = patternToInput(DATA[2].pattern);
      o = calc3Layers(px, res.w, res.b);
      var sample = initGradients3Layer(o, batches[0][0].sign, batches[0][0].input, res.w);
      var averageGradients = sample.init;
      var averageDeltas = [Array(sample.nodeDeltas[0].length).fill(0), Array(sample.nodeDeltas[1].length).fill(0)];
      
      for (var one of batches[y]) {
        o = calc3Layers(one.input, res.w, res.b);
        var grads = initGradients3Layer(o, one.sign, one.input, res.w);
        for (var d of grads.nodeDeltas[0].keys()) {
          averageDeltas[0][d] += grads.nodeDeltas[0][d];
        }
        for (var d of grads.nodeDeltas[1].keys()) {
          averageDeltas[1][d] += grads.nodeDeltas[1][d];
        }
        for (var g of grads.gradients[0].keys()) {
          for (var c of grads.gradients[0][g].keys()) {
            averageGradients[0][g][c] += grads.gradients[0][g][c];
          }
        }
        for (var g of grads.gradients[1].keys()) {
          for (var c of grads.gradients[1][g].keys()) {
            averageGradients[1][g][c] += grads.gradients[1][g][c];
          }
        }
        error += calcError(one.input, res.w, res.b, one.sign, calc3Layers);
        ++rounds;
        ++innerRounds;
      }
      var batchLength = batches[y].length;
      error /= batchLength;

      // update (weights and biases
      res = updateByGradients3Layer(res.w, res.b, averageDeltas, averageGradients, batchLength);
      
      if(!(rounds%2.5e3)){success = true};
      if(success) {
        console.log(`learning round ${rounds} error ${error} k ${k}`);
        success = false;
      }
    }
    if(!(rounds%2.5e3)){success = true};
    if(success) {
      console.log(`learning round ${rounds} error ${error} .k ${k}`);
      success = false;
    }
     ++k;
  }
  return {w: res.w, b: res.b, rounds: rounds}
}