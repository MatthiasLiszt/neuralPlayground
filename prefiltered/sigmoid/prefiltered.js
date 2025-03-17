//settings
const Settings = {
  layers: 2,
  neurons: [30, 10],
  learningRate: 0.01,
  activation: 'sigmoid',
  f: x => 1/(1 + Math.E**(-x))
}

// filter
const xor = (a, b) => (a && !b) || (b && !a); 
const verticalFilter = (q) => xor(q[0] && q[2] , q[1] && q[3]);
const horizontalFilter = (q) => xor(q[0] && q[1] , q[2] && q[3]);
const diagonalFilter = (q) => xor(q[0] && q[3], q[1] && q[2]);
const dotFilter = (q) => (q[0] + q[1] + q[2] + q[3]) == 1;
const cornerFilter = (q) => (q[0] + q[1] + q[2] + q[3]) == 3;
const allFilters = [verticalFilter, horizontalFilter, diagonalFilter, dotFilter, cornerFilter];


// weight bias initialization
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
  var patch = getTrainData(DATA);

  var x = Math.floor(Math.random() * (patch.length + 1)) % patch.length;
  var one = patch[x];
  var input = patternFiltered(one.pattern); 
  
  
  const output = layerCalc(input, weights, bias);

  let errors = [];

  for(let i = 0; i < weights[0].length; ++i) {
    const t = i == one.sign ? 1 : 0.01;
    const oj = output[i];
    const dj = (oj - t) * oj * (1 - oj);
    const error = t - oj;

    for(let j = 0; j < weights[0][0].length; ++j) {
      //weights[0][i][j] += Settings.learningRate * error * input[j];
      weights[0][i][j] -= Settings.learningRate * input[j] * dj;
    }
    bias[0][i] -= Settings.learningRate * dj;
    errors.push(error);
  }

  console.log(`${one.sign}, ${JSON.stringify(input)}`); 
  console.log(`${JSON.stringify(output)}`);
  
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
    var input = patternFiltered(one.pattern); 
    
    const output = layerCalc(input, weights, bias);
    const max = findMax(output);
    const right =  max.at == one.sign; 
    const error = !right ? 1 : 0;
    errors += error;
    console.log(`${one.sign} sign ${max.at} max ${max.max}`);
  }
  return {success: 1 - errors * 0.01, error: errors * 0.01};
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

function patternFiltered(pattern) {
  var i = patternToInput(pattern);

  var square0 = [i[0],i[1],i[3],i[4]];
  var square1 = [i[6],i[7],i[9],i[10]];
  var square2 = [i[9],i[10],i[12],i[13]];
  var square3 = [i[1],i[2],i[4],i[5]];
  var square4 = [i[7],i[8],i[10],i[11]];
  var square5 = [i[10],i[11],i[13],i[14]];

  const squareSeparation = [square0, square1, square2, square3, square4, square5];
  let preFiltered = [];

  for (let filter of allFilters) {
    for (let square of squareSeparation) {
      preFiltered.push(filter(square) ? 1 : 0);
    }
  }
  /*
  for(let one of i) {
    preFiltered.push(one);
  }
  */
  return preFiltered;
}

function getTrainData(data){
  const saturation = 0.8;
  let complete = 0;
  let selected = [];
  let numbers = 0;
  for (let one of data){
    if (one.sign === numbers) {
      selected.push(one);
      ++numbers;
    }
  }
  complete = selected.length / data.length;
  while (complete < saturation) {
    const x = Math.floor(Math.random() * data.length);
    selected.push(data[x]);
    complete = selected.length / data.length;
  }
  return selected;
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
