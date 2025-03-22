// training a spiking neural network to learn a digit

function generateRandomTrainigData(appearance, samples, rightPatterns){
  const data = [];
  for(let i = 0; i < (samples * 0.333) ; ++i) {
    if(Math.random() > appearance){
      data.push(randomSample());
      data.push(randomSample());
      data.push(randomSample());
    } else {
      const x = (Math.random() * rightPatterns.length) % rightPatterns.length;
      data.push(rightPatterns[Math.floor(x)]);
      data.push(rightPatterns[Math.floor(x)]);
      data.push(rightPatterns[Math.floor(x)]); 
    }
  }
  return data;
}

function randomSample() {
  const sample = Math.random() * (2 ** 15 - 1);
  return Math.floor(sample);
}

function initWeights(value, accuracy) {
  accuracy = accuracy === undefined ? 0 : accuracy;
  var Weights = [];
  for(var i = 1; i < Settings.layers; ++i) {
    Weights.push([]);
    for(var j = 0; j < Settings.neurons[i]; ++j) {
      Weights[i - 1].push([]);
      for(var k = 0; k < Settings.neurons[i-1]; ++k) {
        var x = value + Math.random() * accuracy;
        Weights[i - 1][j].push(x);
      }
    }
  }
  return Weights;
}

function initLoads(value) {
  var Loads = [];
  for(var i = 1; i < Settings.layers; ++i) {
    Loads.push([]);
    for(var j = 0; j < Settings.neurons[i]; ++j) {
      Loads[i - 1].push(value);
    }
  }
  return Loads;
}

// important functions

function localMax(input){
// calculates the maximum weight by dividing all the spiking inputs to the weight maximum
  let sum = 0;
  for (let one of input){
    sum += one;
  }
  return Settings.weightMaximum * 1.25 / sum;
}

function localMaxPatch(patch){
// calculcates the maximum weight from all samples in the patch
  let max = 0;
  for(let one of patch) {
    const input = patternToInput(one);
    const local = localMax(input);
    max = local > max ? local : max;
  }
  return max;
}

function checkTrainingRandomly(sampleNumber){
  let rights = 0;
  let falsePositive = 0;
  for(let i = 0; i < sampleNumber; ++i){
    const input = patternToInput(randomSample());
    const output = layerCalc(input, weights, loads);
    falsePositive += output[0] == 1 ? 1 : 0;
  }
  return {falsePositives: falsePositive/sampleNumber * 1e2};
}

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
