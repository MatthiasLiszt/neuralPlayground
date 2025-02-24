// unconventional solution to the xor problem

//settings
const Settings = {
  layers: 2,
  neurons: [4, 1],
  learningRate: 0.01,
  activation: 'stepfunction',
  f: x => x >= 1 ? 1 : 0
}


let weights = initWeights(0.25, 0.1);
let bias = initBias(0.025);

// importat functions

function layerCalc(input, weights, bias) {
  var output = [];
  
  for (var i = 0; i < weights[0].length; ++i) {
    var sum = 0;
    var out = 0;
    for (var j = 0; j < input.length; ++j) {
      const result = input[j] * weights[0][i][j] + bias[0][i];
      //console.log(i + '  ' + j + '  ' +  result );
      sum += result;
    }
    // sum to activiation function which is defined in Settings
    out = Settings.f(sum);
    output.push(out);
  }
  return output;
}

function trainIt() {
  var patch = DATA;

  var x = Math.floor(Math.random() * (patch.length + 1)) % patch.length;
  var one = patch[x];
  var input = [one.pattern & 8 ? 1 : 0, one.pattern & 4 ? 1 : 0, one.pattern & 2 ? 1 : 0, one.pattern & 1 ? 1 : 0];
  
  const output = layerCalc(input, weights, bias);

  let errors = [];

  for(let i = 0; i < weights[0].length; ++i) {
    const x = one.result ? 1 : 0;
    const error = x - output[i];
    for(let j = 0; j < weights[0][0].length; ++j) {
      weights[0][i][j] += Settings.learningRate * error * input[j];
    }
    bias[0][i] += Settings.learningRate * error;
    errors.push(error);
  }

  console.log(`${one.pattern.toString(2)}, ${JSON.stringify(input)}, ${JSON.stringify(errors)}`);
  console.log(`${JSON.stringify(output)}`);
  //console.log(JSON.stringify(weights));
  
  return {output: output, input, errors}
}

function trainX(steps){
  let progress;
  for(let i = 0; i < steps; ++i) {
    progress = trainIt();  
  }
  return progress.errors;
}

function checkTraining(){
  var patch = DATA;
  let errors = 0;
  
  for(let one of patch){
    var input = [one.pattern & 8 ? 1 : 0, one.pattern & 4 ? 1 : 0, one.pattern & 2 ? 1 : 0, one.pattern & 1 ? 1 : 0];
    const output = layerCalc(input, weights, bias);
    const result = one.result ? 1 : 0;
    errors += output[0] == result ? 1 : 0;
    if (output[0] != result) console.log(`${one.pattern.toString(2)} ${output[0] == 1}`); 
  }
  return errors/patch.length;
}

// copy pasted from train.js 

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
