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
    firstLayerDelta.push(sum * oi * (1 - oi));
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

