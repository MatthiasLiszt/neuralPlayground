// singlelayer settings

const Settings = {
  layers: 2,
  neurons: [15, 1],
  learningRate: 0.01,
  weightMaximum: 0.96,
  weightMinimum: 1e-4,
  leak: 0.5,
  activation: 'stepfunction',
  f: x => x >= 1 ? 1 : 0
}

function layerCalc(input, weights, loads) {
  var output = [];
  
  for (var i = 0; i < weights[0].length; ++i) {
    var sum = 0;
    var out = 0;
    for (var j = 0; j < input.length; ++j) {
      const load = input[j] * weights[0][i][j];
      sum += load;
    }
    
    // sum to activiation function which is defined in Settings
    out = Settings.f(sum + loads[0][i]);
    loads[0][i] = out > 0 ? 0 : (loads[0][i] + sum) * Settings.leak;// if sparked membrane/bias is corrected
    output.push(out);
  }

  return output;
}

function trainPatch(patch, rightPatterns) {
  let weightIncrease = 0;
  let weightDecrease = 0;
  let sparks = 0;
  let iteration = 0;  
  let history = {in: [], out: 0};

  // initializing spark history
  for(let one of patternToInput(0)) {
    history.in.push(1);
  }

  const localMaximum = localMaxPatch(rightPatterns);
  let best = {f: 100, i: 0};
  for(let one of patch) {
    const input = patternToInput(one); 
    const output = layerCalc(input, weights, loads);
    //const localMaximum = localMax(input);
    sparks += output[0] == 1 ? 1 : 0;
    //if(output[0] == 1)console.log(`spark at pattern ${one} on iteration ${iteration}`);
    
    if(isNaN(loads[0][0])){
      console.log(`load is NaN at iteration ${iteration}`);
      return NaN;
    }

    let oldcheck = 100;
    const result = trainItSpark(input, output, history, localMaximum);
    const newcheck = checkTrainingRandomly(512); 

    history = result.history;
    weightDecrease += result.weightDecrease;
    weightIncrease += result.weightIncrease;

    oldcheck = newcheck.rights;
    best.i = newcheck.falsePositives < best.f ? iteration : best.i;
    best.f = newcheck.falsePositives < best.f ? newcheck.falsePositives : best.f;
    ++iteration;
  }
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease} sparks ${sparks} cycles ${iteration} best ${best.f} at ${best.i}`);
  console.log(`history ${JSON.stringify(history)}`);
  console.log(`load ${loads[0][0]}`);
  return 0;
}

function trainItSpark(input, output, history, localMaximum){
  let weightDecrease = 0, weightIncrease = 0;
  for(let i = 0; i < weights[0].length; ++i) {
    for(let j = 0; j < weights[0][0].length; ++j) {
      // updating spark history
      history.in[j] = input[j] == 0 ? history.in[j] + 1 : 0;
      history.out = output[0] == 0 ? history.out + 1 : 0;

      const decrease = Settings.learningRate; // * 2 ** -(history.out - history.in[j]);
      const increase = Settings.learningRate * 2 ** -history.in[j];

      // spike time depend plasticity
      const fired = output[0] == 1 || input[j] == 1;

      if(history.out <= history.in[j]){
        weights[0][i][j] = fired && weights[0][i][j] > Settings.weightMinimum ? weights[0][i][j] - decrease : weights[0][i][j];
        weightDecrease += fired && weights[0][i][j] > Settings.weightMinimum ? 1 : 0;
      } else {
        //weights[0][i][j] = fired && weight < Settings.weightMaximum ? weight + increase : weight;
        weights[0][i][j] = fired && weights[0][i][j] < localMaximum ? weights[0][i][j] + increase : weights[0][i][j];
        weightIncrease += fired && weights[0][i][j] < Settings.weightMaximum ? 1 : 0;
      }
      weights[0][i][j] = weights[0][i][j] > localMaximum ? localMaximum : weights[0][i][j];
      weights[0][i][j] = weights[0][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[0][i][j];

    }
  } 
  return {history, weightDecrease, weightIncrease};
}

function trainItHebbian(input, output, localMaximum){
  let weightDecrease = 0, weightIncrease = 0;
  for(let i = 0; i < weights[0].length; ++i) {
    for(let j = 0; j < weights[0][0].length; ++j) {
      // Hebbian Learning
      
      const firedtogether = (input[j] == 1) && (output[0] == 1);
      const fired = (output[0] == 1) || (input[j] == 1);
      const onespark = fired && !firedtogether;

      weightIncrease += firedtogether && (weights[0][i][j] < localMaximum) ? 1 : 0;
      weights[0][i][j] = firedtogether ? weights[0][i][j] + Settings.learningRate : weights[0][i][j];
      weights[0][i][j] = weights[0][i][j] > localMaximum ? localMaximum : weights[0][i][j];
      
      weightDecrease += onespark && weights[0][i][j] > Settings.weightMinimum ? 1 : 0;
      weights[0][i][j] = onespark ? weights[0][i][j] - Settings.learningRate : weights[0][i][j];
      weights[0][i][j] = weights[0][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[0][i][j];
      
    }
  }
  return {weightIncrease, weightDecrease}; 
}

function checkTraining(rightPatterns) {
  for (let one of rightPatterns) {
    const input = patternToInput(one);
    const output = layerCalc(input, weights, loads);
    console.log(`${one} should be one and is ${output[0]}`);
  }
  for (let i = 0; i < 4; ++i) {
    const sample = Math.random() * (2 ** 15 - 1);
    const input = patternToInput(Math.floor(sample));
    const output = layerCalc(input, weights, loads);
    console.log(`${Math.floor(sample)} should be zero and is ${output[0]}`);
  }
}

function checkFullTraining(rightPatterns){
  let rights = 0;
  let falsePositive = 0;
  for(let i = 0; i < 2 ** 15; ++i){
    const input = patternToInput(i);
    const output = layerCalc(input, weights, loads);
    rights += rightPatterns.includes(i) && output[0] == 1 ? 1 : 0;
    falsePositive += output[0] == 1 ? 1 : 0;
  }
  //console.log(`rights ${rights/rightPatterns.length * 1e2} % falsePositives ${falsePositive/(2 ** 15) * 1e2} %`);
  return {rights: rights/rightPatterns.length * 1e2, falsePositives: falsePositive/(2 ** 15) * 1e2};
}