// unconventional solution to the xor problem

//settings
const Settings = {
  layers: 2,
  neurons: [15, 10],
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
  var input = patternToInput(one.pattern); 
  
  
  const output = layerCalc(input, weights, bias);

  let errors = [];

  for(let i = 0; i < weights[0].length; ++i) {
    const x = i == one.sign ? 1 : 0;
    const error = x - output[i];
    for(let j = 0; j < weights[0][0].length; ++j) {
      weights[0][i][j] += Settings.learningRate * error * input[j];
    }
    bias[0][i] += Settings.learningRate * error;
    errors.push(error);
  }

  console.log(`${one.sign}, ${JSON.stringify(input)}, ${JSON.stringify(errors)}`);
  console.log(`${JSON.stringify(output)}`);
  //console.log(JSON.stringify(weights));
  
  return {output: output[one.sign], input, errors}
}

function trainX(steps){
  let progress;
  for(let i = 0; i < steps; ++i) {
    progress = trainIt();
  }
  return progress.errors;
}

function checkTraining(){
  let errors = 0;
  for (let i = 0; i < 1e2; ++i) {
    var patch = DATA;

    var x = Math.floor(Math.random() * (patch.length + 1)) % patch.length;
    var one = patch[x];
    var input = patternToInput(one.pattern); 
    
    const output = layerCalc(input, weights, bias);

    let error = false;
    for (let j = 0; j < 10; ++j) {
      const x = j == one.sign ? 1 : 0;
      error = output[j] == x ? error : true; 
    }
    errors += error ? 1 : 0;
  }
  return errors * 0.01;
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
