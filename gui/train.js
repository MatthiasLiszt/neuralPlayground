
function patternToInput(pattern) {
  var rest = pattern;
  var input = [];
  while(rest > 0) {
    input.push(rest % 2);
    rest *= 0.5;
    rest = Math.floor(rest);
  }
  return input;
}

function initWeights(value) {
  var Weights = [];
  for(var i = 1; i < Settings.layers; ++i) {
    Weights.push([]);
    for(var j = 0; j < Settings.neurons[i]; ++j) {
      Weights[i - 1].push([]);
      for(var k = 0; k < Settings.neurons[i-1]; ++k) {
        Weights[i - 1][j].push(value);
      }
    }
  }
  return Weights;
}

function initBias(value) {
  var Bias = [];
  for(var i = 1; i < Settings.layers; ++i) {
    Bias.push([]);
    for(var j = 0; j < Settings.neurons[i]; ++j) {
      Bias[i - 1].push(value);
    }
  }
  return Bias;
}

function calcLayer(input, weights, bias) {
  var output = [];
  for (var i = 0; i < weights.length; ++i) {
    var sum = 0;
    var out = 0;
    for (var j = 0; j < input.length; ++j) {
      sum += input[j] * weights[i][j] + bias[i];
    }
    // sum to activiation function which is defined in Settings
    out = Settings.f(sum);
    output.push(out);
  }
  return output;
}

function calcFourLayers(input, Weights, Bias) {
  var l1 = calcLayer(input, Weights[0], Bias[0]);
  var l2 = calcLayer(l1, Weights[1], Bias[1]);
  var l3 = calcLayer(l2, Weights[2], Bias[2]);
  return l3;
}

function calc4Layers(input, Weights, Bias) {
  var o1 = calcLayer(input, Weights[0], Bias[0]);
  var o2 = calcLayer(o1, Weights[1], Bias[1]);
  var o3 = calcLayer(o2, Weights[2], Bias[2]);
  return {last: o3, O: [o1, o2, o3]};
}

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

  var lastLayerBiasDelta = [];
  // 2nd layer 
  var secondLayerDelta = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    var sum = 0;
    var oi = outputs.O[1][i];
    
    for (var j = 0; j < Settings.neurons[3]; ++j) {
      var w = Weights[2][j][i];
      sum += w * lastLayerDelta[j];
    }
    /*
    for (var j = 0; j < Settings.neurons[3]; ++j) {
      for (var one of Weights[2][j]) {
        sum += one * lastLayerDelta[j];
      }
    }
    */
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
    /*
    for (var j = 0; j < Settings.neurons[2]; ++j) {
      for (var one of Weights[1][j]) {
        sum += one * secondLayerDelta[j];
      }
    }
    */
    firstLayerDelta.push(sum * oi * (1 - oi));
  }
  //console.log(`last d ${JSON.stringify(lastLayerDelta)}`);
  //console.log(`2nd d ${JSON.stringify(secondLayerDelta)}`);
  //console.log(`1st d ${JSON.stringify(firstLayerDelta)}`);

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

function calcError(input, Weights, Bias, desired) {
  var output = calcFourLayers(input, Weights, Bias);
  // old version
  // return 0.5 * (output[desired] - 1) ** 2;

  var trueValue = 0.7;
  var falseValue = 0.09;

  var sum = 0;
  for (var i = 0; i < 10; ++i) {
    var value = 0;
    value = i == desired ? 0.5 * (output[i] - trueValue) ** 2 : 0.5 * (output[i] - falseValue) ** 2;
    sum += value;
  } 
  return sum;
} 

function learnOne(steps) {
  var w = initWeights(0.1);
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
  DATA.length = 4;
  var w = initWeights(0.1);
  var b = initBias(0.1);
  var sample = DATA[2];
  var px = patternToInput(sample.pattern);
  
  var o = calc4Layers(px, w, b);
  var res = backprop4Layer(o, w, b, sample.sign, px);

  var rounds = 0;
  var success = false;
  for (var k = 0; k < 1024; ++k) {
    for (var one of DATA) {
      var p = patternToInput(one.pattern);
      while(calcError(p, res.w, res.b, one.sign) > 0.07  ) {
        o = calc4Layers(p, res.w, res.b);
        res = backprop4Layer(o, res.w, res.b, one.sign, p);
        ++rounds;
        if(!(rounds%2.5e4)) success = true;
      }
      if(success) console.log(`learning round ${rounds}`);
      success = false;
    }
  }

  var right = 0;
  for (let one of DATA) {
    var p = patternToInput(one.pattern);
    o = calcFourLayers(p, res.w, res.b);
    if (one.sign == findMax(o).at ) {
      showFinalLayers(o);
      console.log(`brain recognized ${one.sign} +`);
      var e3 = calcError(px, res.w, res.b, 3);
      var ex = calcError(p, res.w, res.b, one.sign);
      console.log(`error for ${sample.sign} ${e3} , error for ${one.sign} ${ex}`);
      ++right;
    } 
  }

  return {w: res.w, b: res.b, p: px, right: right/DATA.length, rounds: rounds};
}


function further(data){
  var w = data.w;
  var b = data.b;
  var p = data.p;
  var l = randomLearn(p, w, b, 1, 8e4);
  var o = calcFourLayers(p, l.w, l.b);
  console.log(JSON.stringify(o));
  return data;
}

function dumpWeights(weights){
  for(var i = 0; i < weights.length; ++i) {
    for(var j = 0; j < weights[i].length; ++j) {
      //console.log(weights[i][j].join('  '));
      var line = [];
      for (var one of weights[i]) {
        var n = one.toString();
        line.push(n.substring(0,8));
      }
      console.log(line.join('  '));
    }
  }
}

function findMax(field) {
  var max = field[0];
  var at = 0;
  var index = 0;
  for(var one of field) {
    if(one > max) {
      at = index;
      max = one;
    }
    ++index;
  }
  return {max: max, at: at};
}

function showFinalLayers(data) {
  var format = [];
  for(var i = 0; i < data.length; ++i) {
    var s = data[i].toString();
    format.push(`(${i}) ${s.substring(0,6)}`);
  }
  console.log(format.join('  '));
}
