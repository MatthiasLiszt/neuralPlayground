function calcAllError(Weights, Bias) {
  var sum = 0;
  for(var i = 0; i < DATA.length; ++i) {
    var p = patternToInput(DATA[i].pattern);
    sum += calcError(p, Weights, Bias, DATA[i].sign);
  }
  return sum;
}

function randomChange(Weights) {
  var Changed = JSON.parse(JSON.stringify(Weights));
  var layer = Math.floor(Math.random() * (Settings.layers - 1));
  var neuron = Math.floor(Math.random() * Settings.neurons[layer + 1]);
  var index = Math.floor(Math.random() * Settings.neurons[layer]);
  var rate = Settings.learningRate;
  var change = Math.random() > 0.5 ? 1 - rate : 1 + rate;
  Changed[layer][neuron][index] *= change;
  // console.log(layer + ' ' + neuron + ' ' + Changed[layer][neuron][index] + ' change ' + change);
  return Changed;
}

function randomChangeBias(Bias) {
  var Changed = JSON.parse(JSON.stringify(Bias));
  var layer = Math.floor(Math.random() * (Settings.layers - 1));
  var index = Math.floor(Math.random() * Settings.neurons[layer]);
  var rate = Settings.learningRate;
  var change = Math.random() > 0.5 ? 1 - rate : 1 + rate;
  Changed[layer][index] *= change;
  return Changed;
}

function randomLearn(input, Weights, Bias, desired, steps) {
  var best = 0;
  for (var i = 0; i < steps; ++i) {
    var Changed = randomChange(Weights);
    var oldError = calcError(input, Weights, Bias, desired);
    var newError = calcError(input, Changed, Bias, desired);
    var progress = newError < oldError ? Math.abs((oldError / newError) - 1) : 1;
    //if (progress < Settings.learningRate && progress > Settings.learningRate ** 3) console.log('!!! ' + progress);
    if (progress < Settings.learningRate && progress > best) {
      //console.log('!!! ' + progress);
      best = progress;
    }
    Weights = progress < Settings.learningRate && progress > Settings.learningRate ** 3 ? Changed : Weights;
  }
  return {w: Weights, b: Bias};
}

function testRandomFirst() {
  var w = initWeights(0.1);
  var b = initBias(0);
  var p = patternToInput(DATA[0].pattern);
  var l = randomLearn(p, w, b, 1, 4e4);
  var o = calcFourLayers(p, l.w, l.b);
  console.log(JSON.stringify(o));
  return {w: l.w, b: l.b, p: p};
}

function testRandomAll() {
  var w = initWeights(0.1);
  var b = initBias(0);
  var p1 = patternToInput(DATA[0].pattern);
  //var p9 = patternToInput(DATA[9].pattern);
  var l = [];
  for (var i = 0; i < DATA.length; ++i) {
    var p = patternToInput(DATA[i].pattern);
    l = randomLearn(p, w, b, DATA[i].sign, 4e4);
    console.log('.');
    w = l.w;
    b = l.b;
  }
  var o = calcFourLayers(p1, l.w, l.b);
  console.log(JSON.stringify(o));
  //o = calcFourLayers(p9, l.w, l.b);
  //console.log(JSON.stringify(o));
  return {w: l.w, b: l.b, p: p};
}

function randomAllLearn(Weights, Bias, steps) {
  var preError = calcAllError(Weights, Bias);
  for (var i = 0; i < steps; ++i) {
    var Changed = randomChange(Weights);
    var nBias = randomChangeBias(Bias);
    var oldError = calcAllError(Weights, Bias);
    var newError = calcAllError(Changed, nBias);
    //var newError = calcAllError(Changed, Bias);
    if (!(i % 5000) && newError < oldError) {
      console.log('!!! ' + newError + ' at step ' + i);
    }
    Weights = newError > (1 - Settings.learningRate) * oldError && newError < oldError ? Changed : Weights;
    nBias = newError > (1 - Settings.learningRate) * oldError && newError < oldError ? nBias : Bias;
    if (i == steps - 2) console.log('final ' + newError);
  }
  return {w: Weights, b: Bias, preError: preError};
}

function testRandomIt() {
  DATA.length = 2;
  //Settings.learningRate = 0.001;
  var w = initWeights(0.1);
  var b = initBias(0.05);
  var p0 = patternToInput(DATA[0].pattern);
  var l = randomAllLearn(w, b, 1e6);
  
  var right = 0;
  for (let one of DATA) {
    var p = patternToInput(one.pattern);
    var o = calcFourLayers(p, l.w, l.b);
    if (one.sign == findMax(o).at ) {
      console.log(JSON.stringify(o));
      console.log(`brain recognized ${one.sign} +`);
      ++right;
    }
  }
  console.log(`brain got ${right} of ${DATA.length} right`);
  return {w: l.w, b: l.b, p: p0, right: right / DATA.length};
}
