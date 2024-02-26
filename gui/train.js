var DATA = [];
var Cache = {};
var Mode = 'idle';

//version for just 15 input neurons
function patternToInput(pattern) {
  var rest = pattern;
  var input = [];
  while(rest > 0) {
    input.push(rest % 2);
    rest *= 0.5;
    rest = Math.floor(rest);
  }
  while(input.length < 15) {
    input.push(0);
  }
  return input;
}

function initWeights(value, accuracy) {
  accuracy = accuracy === undefined ? 0 : accuracy;
  var Weights = [];
  for(var i = 1; i < Settings.layers; ++i) {
    Weights.push([]);
    for(var j = 0; j < Settings.neurons[i]; ++j) {
      Weights[i - 1].push([]);
      for(var k = 0; k < Settings.neurons[i-1]; ++k) {
        Weights[i - 1][j].push(value + Math.random() * accuracy);
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

function calc3Layers(input, Weights, Bias) {
  var o1 = calcLayer(input, Weights[0], Bias[0]);
  var o2 = calcLayer(o1, Weights[1], Bias[1]);
  return {last: o2, O: [o1, o2]};
}

function calcError(input, Weights, Bias, desired, calcAllLayers) {
  var output = calcAllLayers(input, Weights, Bias).last;
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

function checkLearningResults(WeightsBias, patch) {
  var res = WeightsBias;
  var right = 0;
  var calcAllLayers = Settings.layers == 3 ? calc3Layers : calc4Layers;
  for (let one of patch) {
    var p = patternToInput(one.pattern);
    o = calcAllLayers(p, res.w, res.b).last;
    if (one.sign == findMax(o).at ) {
      showFinalLayers(o);
      console.log(`brain recognized ${one.sign}`);
      var ex = calcError(p, res.w, res.b, one.sign, calcAllLayers);
      console.log(`error for ${one.sign} ${ex}`);
      ++right;
    } 
  }
  return right;
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
    var s = data[i] < 1e-6 ? '0' : data[i].toString();
    format.push(`(${i}) ${s.substring(0,6)}`);
  }
  console.log(format.join('  '));
}

function testBy(method, calcAllLayers) {
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var p5 = patternToInput(DATA[5].pattern);
  p5 = p5.map(x => x = x == 0 ? 0.001 : x);
  //var o = calc4Layers(p5, w, b);
  var o = calcAllLayers(p5, w, b);
  console.log(JSON.stringify(o));
  var sign = DATA[5].sign;
  var res = method(o, w, b, sign, p5);
  // dumpWeights(res.w);
  var error = calcError(p5, res.w, res.b, sign, calcAllLayers);
  return {w: res.w, b: res.b, p: p5, error: error, ld: res.ld};
}

function testXstepsBy(method, calcAllLayers, steps) {
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var p5 = patternToInput(DATA[5].pattern);
  p5 = p5.map(x => x = x == 0 ? 0.001 : x);
  var sign = DATA[5].sign;
  var o = calcAllLayers(p5, w, b);
  var res = method(o, w, b, sign, p5);
  var error = calcError(p5, res.w, res.b, sign, calcAllLayers);
  var best = 1e3;

  for (var i = 0; i < steps; ++i) {
    //o = calc4Layers(p5, w, b);
    o = calcAllLayers(p5, w, b);
    var hardCopy = JSON.parse(JSON.stringify(res));
    res = method(o, res.w, res.b, sign, p5);
    error = calcError(p5, res.w, res.b, sign, calcAllLayers);
    if (error < best) {
      best = error;
    } else {
      res = hardCopy;
    }
  }

  error = calcError(p5, res.w, res.b, sign, calcAllLayers);
  return {w: res.w, b: res.b, p: p5, error: error, ld: res.ld, best: best};
}

function testItBy(method) {
  var patch = DATA;
  var w = initWeights(0.25, 0.1);
  var b = initBias(0.025);
  var sample = DATA[2];
  var px = patternToInput(sample.pattern);
  
  var calcAllLayers = Settings.layers == 3 ? calc3Layers : calc4Layers;
  var o = calcAllLayers(px, w, b);
  var res = method(o, w, b, sample.sign, px);

  // accuracy parameter seems to be critical for the result -- so far 0.07 does well
  res = trainPatchRandomlyBy(res, patch, 0.152, method, 6e4);
  var rounds = res.rounds;

  var right = checkLearningResults(res, patch);

  return {w: res.w, b: res.b, p: px, right: right/patch.length, rounds: rounds, patch: patch};
}

function trainPatchRandomlyBy(WeightsBias, patch, accuracy, method, maxUnit) {
  var res = WeightsBias;
  var rounds = 0;
  var success = false;
  var k = 0;
  var calcAllLayers = Settings.layers == 3 ? calc3Layers : calc4Layers;
  while (k < 1e4 * patch.length) {
    var x = Math.floor(Math.random() * (patch.length + 1)) % patch.length;
    var one = patch[x];
    var p = patternToInput(one.pattern);
    var innerRounds = 0;
    while(calcError(p, res.w, res.b, one.sign, calcAllLayers) > accuracy && innerRounds < maxUnit) {
      o = calcAllLayers(p, res.w, res.b);
      res = method(o, res.w, res.b, one.sign, p);
      ++rounds;
      ++innerRounds;
      if(!(rounds%2.5e4)){success = true};
      if(success) {
        console.log(`learning round ${rounds} error ${calcError(p, res.w, res.b, one.sign, calcAllLayers)} k/patch.length ${k/patch.length}`);
        success = false;
      }
    }
    if(!(rounds%2.5e4)){success = true};
    if(success) {
      console.log(`learning round ${rounds} error ${calcError(p, res.w, res.b, one.sign, calcAllLayers)} k/patch.length ${k/patch.length}`);
      success = false;
    }
     ++k;
  }
  return {w: res.w, b: res.b, rounds: rounds}
}

function loadDefaultTrainingData() {
  var id = document.getElementById('dataCount');
  DATA = DEFAULT;
  id.textContent = `${DATA.length} entries`;
}