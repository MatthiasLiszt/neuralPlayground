// unconventional solution to the xor problem

//settings
const Settings = {
  layers: 2,
  neurons: [2, 1],
  learningRate: 0.01,
  //activation: 'sigmoid',
  //f: x => 1/(1 + Math.E**(-x))
  activation: 'threshold',
  f: x => x >= 1 ? 1 : 0
}

const Distinction = 0.5;
const normalize = (x) => x = x > Distinction ? 1 : 0;
const xor = (a, b) => normalize(a) ^ normalize(b) ? 1 : 0;
const and = (a, b) => normalize(a) || normalize(b) ? 1 : 0;
const nor = (a, b) => !(normalize(a) && normalize(b)) ? 1 : 0;
let goal = nor;

let weights = initWeights(0.25, 0.1);
let bias = initBias(0.025);
//let bias = initBias(0);

let Highscore = [{score: 0, weights: '[[[0,0,0]]]', i: 0, t: 0},
                 {score: 0, weights: '[[[0,0,0]]]', i: 1, t: 1e3},
                 {score: 0, weights: '[[[0,0,0]]]', i: 2, t: 2e3},
                 {score: 0, weights: '[[[0,0,0]]]', i: 3, t: 4e3}];

// solving XOR problem

let progress = 0;
goal = and;
progress = trainX(1e4);
const And = {weights: JSON.parse(JSON.stringify(weights)), bias: JSON.parse(JSON.stringify(bias)), progress: progress.progress};
goal = nor;
progress = trainX(1e4);
const Nor = {weights: JSON.parse(JSON.stringify(weights)), bias: JSON.parse(JSON.stringify(bias)), progress: progress.progress};
progress = trainXorS(4e4);

console.log('NOR ' + JSON.stringify(Nor));
console.log('AND ' + JSON.stringify(And));
console.log('progress XOR ' + progress.progress);

function trainXor() {
  const a = normalize(Math.random());
  const b = normalize(Math.random());
  
  const output1 = and(a, b);
  const output2 = nor(a, b);

  const input = [output1, output2];

  const output = layerCalc(input, weights, bias);
  const result = xor(a,b);

  //const error = result == normalize(output[0]) ? 0 : result - output[0];
  error = result - output[0];

  for(let i = 0; i < weights.length; ++i) {
    weights[0][0][i] += Settings.learningRate * error * input[i];
  }

  bias[0][0] += Settings.learningRate * error;

  console.log(`${output[0]}, ${JSON.stringify(input)}, ${result}, ${error}`);
  console.log(JSON.stringify(weights));
  return {output: output[0], input, result, error}
}

function trainXorS(steps){
  let hit = 0;
  for(let i = 0; i < steps; ++i) {
    const result = trainXor();
    hit = result.error == 0 ? ++hit : hit;
  }
  return {progress: hit/steps }
}

// importat functions

function layerCalc(input, weights, bias) {
  var output = [];
  for (var i = 0; i < weights.length; ++i) {
    var sum = 0;
    var out = 0;
    for (var j = 0; j < input.length; ++j) {
      const result = input[j] * weights[0][0][i] + bias[0][0];
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
  const a = normalize(Math.random());
  const b = normalize(Math.random());

  const input = [a, b];
  const output = layerCalc(input, weights, bias);
  const result = goal(a, b);

  //const error = result == normalize(output[0]) ? 0 : result - output[0];
  error = result - output[0];

  for(let i = 0; i < weights.length; ++i) {
    weights[0][0][i] += Settings.learningRate * error * input[i];
  }

  bias[0][0] += Settings.learningRate * error;

  console.log(`${output[0]}, ${JSON.stringify(input)}, ${result}, ${error}`);
  console.log(JSON.stringify(weights));
  return {output: output[0], input, result, error}
}

function trainX(steps){
  let hit = 0;
  for(let i = 0; i < steps; ++i) {
    const result = trainIt();
    hit = result.error == 0 ? ++hit : hit;
    if (i > 1e3) {
      toHighscore(weights, hit/i, i);
    }
  }
  return {progress: hit/steps }
}

function toHighscore(weights, score, step){
  let change = true;
  let round = x => Math.floor(x * 4) / 4;
   
  for(let one of Highscore) {
    let parsed = JSON.parse(one.weights);
    let distinct1 = round(weights[0][0][0]) != round(parsed[0][0][0]);
    let distinct2 = round(weights[0][0][1]) != round(parsed[0][0][1]);
    let distinct3 = Math.abs(one.score - score) > 0.05;
    let distinct4 = Math.abs(one.t - step) > 1e3; 
    let different = distinct1 && distinct2 && distinct3 && distinct4;
    
    if(score > one.score && change && different) {
      Highscore[one.i].score = score;
      Highscore[one.i].weights = JSON.stringify(weights);
      Highscore[one.i].t = step;
      change = false;
    }
  }
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

