// code and settings for the neural net

function generateRandomTrainingData(appearance, samples, rightPatterns){
  const data = [];
  const N = Settings.neurons[0];
  for(let i = 0; i < (samples * 0.333) ; ++i) {
    if(Math.random() > appearance){
      data.push(randomSample(N));
      data.push(randomSample(N));
      data.push(randomSample(N));
    } else {
      const x = Math.random() * (rightPatterns.length - 1);
      data.push(rightPatterns[Math.floor(x)].flat());
      data.push(rightPatterns[Math.floor(x)].flat());
      data.push(rightPatterns[Math.floor(x)].flat()); 
    }
  }
  return data;
}

function randomSample(N) {
  const sample = Math.floor(Math.random() * (2 ** N - 1));
  let image = [];
  for(let i = 0; i < N; ++i){
    image.push(sample%(2**i) == 0 ? 1 : 0);
  }
  return image;
}

//settings
const Settings = {
  layers: 3,
  neurons: [16, 12, 1],
  learningRate: 0.01,
  weightMaximum: 0.96,
  weightMinimum: 1e-4,
  activation: 'stepfunction',
  f: x => x >= 1 ? 1 : 0
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

function calcLayer(input, weights) {
  var output = [];
  for (var i = 0; i < weights.length; ++i) {
    var sum = 0;
    var out = 0;
    for (var j = 0; j < input.length; ++j) {
      sum += input[j] * weights[i][j];
      if(isNaN(weights[i][j])){
        console.log(`weight is ${weights[i][j]} in calcLayer, i ${i}, j${j}`);
        return undefined;
      }
    }
    // sum to activiation function which is defined in Settings
    out = Settings.f(sum);
    output.push(out);
  }
  return output;
}

function calc3Layers(input, Weights) {
  var o1 = calcLayer(input, Weights[0]);
  var o2 = calcLayer(o1, Weights[1]);
  return {last: o2, O: [o1, o2]};
}

function layerCalc(input, weights) {
  const result = calc3Layers(input,weights);
  return {mid: result.O[0], final: result.O[1]};
}


function localMax(input){
  // calculates the maximum weight by dividing all the spiking inputs to the weight maximum
    let sum = 0;
    for (let one of input){
      sum += one;
    }
    return Settings.weightMaximum * 1.25 / sum;
}

function trainPatchCompetitiveOnly(patch) {
  let weightIncrease = 0;
  let weightDecrease = 0;
 
  for(let one of patch) {
    const localMaximum = Settings.weightMaximum; 

    let result = {weightDecrease, weightIncrease};
    const hidden = layerCalc(one, weights).mid;
    
    result = trainNeuron(one, hidden, localMaximum, hidden.indexOf(1) >= 0 ? hidden.indexOf(1) : 0);
 
    weightDecrease += result.weightDecrease;
    weightIncrease += result.weightIncrease;
  }
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease}`);
 
  weights[1][0] = [1,1,1, 1,1,1, 1,1,1];
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease}`);
 
  const culled = cullHiddenNeurons(PATTERNS);
  if(culled == 1)console.log(`training failed`);
  return culled;
}

function trainPatchCompetitivePerceptron(patch, rightPatterns) {
  let weightIncrease = 0;
  let weightDecrease = 0;
 
  for(let one of patch) {
    const localMaximum = Settings.weightMaximum; 

    let result = {weightDecrease, weightIncrease};
    const hidden = layerCalc(one, weights).mid;
    
    result = trainNeuron(one, hidden, localMaximum, hidden.indexOf(1) >= 0 ? hidden.indexOf(1) : 0);
 
    weightDecrease += result.weightDecrease;
    weightIncrease += result.weightIncrease;
  }
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease}`);

  let RightPatterns = [];
  for(let one of rightPatterns){
    RightPatterns.push(one.flat().join());
  }

 let fired = 0;
  for(let one of patch) {
    const all = layerCalc(one, weights);

    // perceptron learning rule 
    const right = RightPatterns.includes(one.join()) ? 1 : 0;
    const error = right - all.final[0];
    trainNeuronPerceptron(all.mid, error, 0, 1);

    fired += all.final[0];
  }
  console.log(`fired ${fired}`);
}

function trainNeuron(pattern, output, localMaximum, neuron){
  const input = pattern;
  const i = neuron;

  let weightDecrease = 0, weightIncrease = 0;

  for(let j = 0; j < weights[0][0].length; ++j) {
    const firedtogether = input[j] == 1 && output[0] == 1;
    const fired = input[j] == 1 || output[0] == 1;
    const onespark = fired && !firedtogether;

    weightIncrease += firedtogether ? 1 : 0;
    weightDecrease += onespark ? 1 : 0;

    weights[0][i][j]= firedtogether ? weights[0][i][j] + Settings.learningRate : weights[0][i][j];
    weights[0][i][j]= onespark ? weights[0][i][j] - Settings.learningRate : weights[0][i][j];

    weights[0][i][j] = weights[0][i][j] > localMaximum ? localMaximum : weights[0][i][j];
    weights[0][i][j] = weights[0][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[0][i][j];
  }

  return {weightDecrease, weightIncrease};
}

function checkTrainingRandomly(sampleNumber, neuron){
  let falsePositive = 0;
  const N = Settings.neurons[0];
  for(let i = 0; i < sampleNumber; ++i){
    const input = randomSample(N);
    const output = calcSimpleNeuron(input, weights[0][neuron]);
    falsePositive += output == 1 ? 1 : 0;
  }
  return {falsePositives: falsePositive/sampleNumber};
}

function calcSimpleNeuron(input, Weights){
  var sum = 0;
  var out = 0;
  for (var j = 0; j < input.length; ++j) {
    sum += input[j] * Weights[j];
  }
  // sum to activiation function which is defined in Settings
  out = Settings.f(sum);
  
  return out;
}

function cullHiddenNeurons(rightPatterns, falseRate = 1){
  let culled = 0;
  for(let n = 0; n < Settings.neurons[1]; ++n) {
    let rights = 0;
    const falsePositives = checkTrainingRandomly(1024, n).falsePositives;
    for(let one of rightPatterns){
      const output = calcSimpleNeuron(one.flat(), weights[0][n]);
      rights += output;
    }
    rights = rights / rightPatterns.length;
    console.log(`neuron ${n} false ${falsePositives} right ${rights} condition ${(rights == 0 || falsePositives > falseRate)}`)
    if(rights == 0 || falsePositives > falseRate){
      for(let i = 0; i < weights[0][n].length; ++i){
        weights[0][n][i] = 0;
      }
      ++culled;
    }   
  }
  return culled/Settings.neurons[1];
}

function trainNeuronPerceptron(input, error, neuron, layer){
  const i = neuron;
  const l = layer;

  for(let j = 0; j < weights[layer][i].length; ++j) {
    weights[l][i][j] += Settings.learningRate * error * input[j];
    
    weights[l][i][j] = weights[l][i][j] > Settings.weightMaximum ? Settings.weightMaximum : weights[l][i][j];
    weights[l][i][j] = weights[l][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[l][i][j];
  }
}

function createDeficiencies(coverage = 0.9){
  const x = GRID[0].length;
  const xx = x * x;
  let deficiencies = [];
  let history = [];
  let complete = Array(xx).fill(0);
  let completeness = 0;
  let iteration = 0;
  console.log('create deficiencies ' + x);
  
  while (iteration < 1e2 && completeness < coverage){
    deficiencies = [];
    for(let i = 0; i < x; ++i){
      let sample = Array(xx).fill(0);
      let start = Math.floor(Math.random() * x);
      let half = Math.random() < 0.5 ? 0 : sample.length * 0.5;
      while (history.includes(start + half) && history.length < xx){
        start = Math.floor(Math.random() * x);
        half = Math.random() < 0.5 ? 0 : sample.length * 0.5;
      }
      history.push(start + half);
      history.push(start + 1 + half);
      history.push(start + half + x);
      history.push(start + 1 + half + x);
      sample[(start + half) % xx] = 1;
      sample[(start + 1 + half) % xx] = 1;
      sample[(start + half + x) % xx] = 1;
      sample[(start + 1 + half + x) % xx] = 1;
      deficiencies.push(sample); 
    }
    console.log(JSON.stringify(deficiencies));
    complete = Array(xx).fill(0);
    completeness = 0;
    for(let one of deficiencies){
      for(let i = 0; i < xx; ++i){
        complete[i] = one[i] == 1 ? 1 : complete[i];
      }
    }
    for (let one of complete){
      completeness += one;
    }
    completeness = completeness / xx;
    console.log(complete.join() + ' completeness ' + completeness);
    ++iteration;
  }
  return deficiencies;
}

function trainPatchPerceptronOnly(patch, rightPatterns, deficiencies) {
  console.log(JSON.stringify(weights));

  let RightPatterns = [];
  for(let one of rightPatterns){
    RightPatterns.push(one.flat().join());
  }

  let fired = 0;
  let errors = 0;

  for(let one of patch) {
    const hidden = layerCalc(one, weights).mid;
    for (let i = 0; i < hidden.length; ++i){
      const right = RightPatterns.includes(one.join()) ? 1 : 0;
      const error = right - hidden[i];
      fired += hidden[i] == 1 ? 1 : 0;
      errors += error != 0 ? 1 : 0;
      const deficiency = deficiencies[i % GRID[0].length];
      weights[0][i] = doFilter(weights[0][i], deficiency);
      trainNeuronPerceptron(doFilter(one, deficiency), error, i, 0);
    }
  }

  for(let one of patch) {
    const all = layerCalc(one, weights);

    // perceptron learning rule 
    const right = RightPatterns.includes(one.join()) ? 1 : 0;
    const error = right - all.final[0];
    trainNeuronPerceptron(all.mid, error, 0, 1);

    fired += all.final[0];
  }
  console.log(`fired ${fired} errors ${errors}`);
  console.log(JSON.stringify(weights));
}

function doFilter(input, deficiency){
  let output = [];
  for(let i = 0; i < input.length; ++i){
    output.push(deficiency[i] == 1 ? input[i] : 0);
  }
  return output;
}