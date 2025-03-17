// training a spiking neural network to learn a digit

function generateRandomTrainigData(appearance, samples, rightPatterns){
  const data = [];
  for(let i = 0; i < (samples * 0.333) ; ++i) {
    if(Math.random() > appearance){
      data.push(randomSample());
      data.push(randomSample());
      data.push(randomSample());
    } else {
      const x = Math.random() * (rightPatterns.length - 1);
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

//settings
const Settings = {
  layers: 2,
  neurons: [15, 1],
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

// important functions

function layerCalc(input, weights) {
  var output = [];
  
  for (var i = 0; i < weights[0].length; ++i) {
    var sum = 0;
    var out = 0;
    for (var j = 0; j < input.length; ++j) {
      const load = input[j] * weights[0][i][j];
      sum += load;
    }

    // sum to activiation function which is defined in Settings
    out = Settings.f(sum);
    output.push(out);
  }
  return output;
}

function localMax(input){
// calculates the maximum weight by dividing all the spiking inputs to the weight maximum
  let sum = 0;
  for (let one of input){
    sum += one;
  }
  return Settings.weightMaximum * 2.5 / sum;
}

// | *******
// |  WARNING THE TRAINING IS EXTREMELY UNSTABLE AT MIGHT REQUIRE UP TO 20 ATTEMPTS !!!
// | *******
function trainPatch(patch) {
  let weightIncrease = 0;
  let weightDecrease = 0;
  let sparks = 0;
  let iteration = 0;

  for(let one of patch) {
    const input = patternToInput(one); 
    const output = layerCalc(input, weights);
    const localMaximum = localMax(input);
    sparks += output[0] == 1 ? 1 : 0;
    //if(output[0] == 1)console.log(`spark at pattern ${one} on iteration ${iteration}`);

    ++iteration;
    for(let i = 0; i < weights[0].length; ++i) {
      for(let j = 0; j < weights[0][0].length; ++j) {
        const firedtogether = input[j] == 1 && output[0] == 1;
        const fired = input[j] == 1 || output[0] == 1;
        const onespark = fired && !firedtogether;

        weightIncrease += firedtogether ? 1 : 0;
        weights[0][i][j]= firedtogether && weights[0][i][j] < localMaximum ? weights[0][i][j] + Settings.learningRate : weights[0][i][j];
        weights[0][i][j] = weights[0][i][j] > Settings.weightMaximum ? Settings.weightMaximum : weights[0][i][j];

        weightDecrease += onespark ? 1 : 0;
        weights[0][i][j]= onespark ? weights[0][i][j] - Settings.learningRate : weights[0][i][j];
        weights[0][i][j] = weights[0][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[0][i][j];
      }
    } 
  }
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease} sparks ${sparks} cycles ${iteration}`);
}

function checkTraining(rightPatterns) {
  for (let one of rightPatterns) {
    const input = patternToInput(one);
    const output = layerCalc(input, weights);
    console.log(`${one} should be one and is ${output[0]}`);
  }
  for (let i = 0; i < 4; ++i) {
    const sample = Math.random() * (2 ** 15 - 1);
    const input = patternToInput(Math.floor(sample));
    const output = layerCalc(input, weights);
    console.log(`${Math.floor(sample)} should be zero and is ${output[0]}`);
  }
}

function checkFullTraining(rightPatterns){
  let rights = 0;
  let falsePositive = 0;
  for(let i = 0; i < 2 ** 15; ++i){
    const input = patternToInput(i);
    const output = layerCalc(input, weights);
    rights += rightPatterns.includes(i) && output[0] == 1 ? 1 : 0;
    falsePositive += output[0] == 1 ? 1 : 0;
  }
  //console.log(`rights ${rights/rightPatterns.length * 1e2} % falsePositives ${falsePositive/(2 ** 15) * 1e2} %`);
  return {rights: rights/rightPatterns.length * 1e2, falsePositives: falsePositive/(2 ** 15) * 1e2};
}

function optimizingWeights(rightPatterns){
  let average = 0;
  for(let i = 0; i < weights[0].length; ++i) {
    for(let j = 0; j < weights[0][0].length; ++j) {
      average += weights[0][i][j];
    }
  }
  average = average / weights[0][0].length;
  console.log(`average ${average}`);
  let decrease = 0;
  let changes = 0;
  let rights = 100;
  let backup = [];
  do {
    changes = 0;
    backup = JSON.parse(JSON.stringify(weights));
    let falsePositives = checkFullTraining(rightPatterns).falsePositives;
    for(let i = 0; i < weights[0].length; ++i) {
      for(let j = 0; j < weights[0][0].length; ++j) {
        weights[0][i][j] = weights[0][i][j] > average ? weights[0][i][j] - weights[0][i][j] * Settings.learningRate : weights[0][i][j];
        changes += weights[0][i][j] > average ? 1 : 0;
      }
    }
    const results = checkFullTraining(rightPatterns);
    rights = results.rights;
    decrease = results.falsePositives - falsePositives;
    console.log(`falsePositives ${falsePositives} decrease ${decrease} changes ${changes}`);
  } while (decrease <= 0 && changes > 0 && rights == 100);
  return backup;
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
