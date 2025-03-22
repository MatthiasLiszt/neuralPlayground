//settings
const Settings = {
  layers: 3,
  neurons: [15, 9, 1],
  learningRate: 0.01,
  weightMaximum: 1.41,
  weightMinimum: 1e-4,
  leak: 0.5,
  activation: 'stepfunction',
  f: x => x >= 1 ? 1 : 0
}

function calc3Layers(input, Weights, Loads) {
  var o1 = calcLayer(input, Weights[0], Loads[0]);
  var o2 = calcLayer(o1, Weights[1], Loads[1]);
  return {last: o2, O: [o1, o2]};
}

function calcLayer(input, weights, loads) {
  var output = [];
  //console.log(`${JSON.stringify(weights)}`);
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
    out = Settings.f(sum + loads[i]);
    //console.log(`... ${sum}`);
    loads[0][i] = out > 0 ? 0 : (loads[i] + sum) * Settings.leak;// if sparked membrane/bias is corrected
    output.push(out);
  }
  return output;
}

function calcSimpleNeuron(input, weights, load){
  var sum = 0;
  var out = 0;
  for (var j = 0; j < input.length; ++j) {
    sum += input[j] * weights[j];
    if(isNaN(weights[j])){
      console.log(`weight is ${weights[j]} in calcLayer j${j}`);
      return undefined;
    }
  }
  // sum to activiation function which is defined in Settings
  out = Settings.f(sum + load);
  load = out > 0 ? 0 : (load + sum) * Settings.leak;// if sparked membrane/bias is corrected
  
  return out;
}

function layerCalc(input, weights, loads) {
  const result = calc3Layers(input,weights, loads);
  return {mid: result.O[0], final: result.O[1]};
}

function trainItSparkLayer1(input, hidden, history, localMaximum){
  let weightDecrease = 0, weightIncrease = 0;
  for (let i = 0; i < weights[0].length; ++i) {
    for(let j = 0; j < weights[0][0].length; ++j) {
      // updating spark history
      history.in[j] = input[j] == 0 ? history.in[j] + 1 : 0;
      history.mid[i] = hidden[i] == 0 ? history.mid[i] + 1 : 0;

      const decrease = Settings.learningRate; 
      const increase = Settings.learningRate * 2 ** -history.in[j];

      // spike time depend plasticity
      const fired = hidden[i] == 1 || input[j] == 1;

      if(history.mid[i] <= history.in[j]){
        weights[0][i][j] = fired && weights[0][i][j] > Settings.weightMinimum ? weights[0][i][j] - decrease : weights[0][i][j];
        weightDecrease += fired && weights[0][i][j] > Settings.weightMinimum ? 1 : 0;
      } else {
        weights[0][i][j] = fired && weights[0][i][j] < localMaximum ? weights[0][i][j] + increase : weights[0][i][j];
        weightIncrease += fired && weights[0][i][j] < Settings.weightMaximum ? 1 : 0;
      }
      weights[0][i][j] = weights[0][i][j] > localMaximum ? localMaximum : weights[0][i][j];
      weights[0][i][j] = weights[0][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[0][i][j];

    }
  }
  return {history, weightDecrease, weightIncrease};
}

function trainNeuron(pattern, hidden, history, localMaximum, neuron, ignorePattern){
  const input = patternToInput(pattern);
  let weightDecrease = 0, weightIncrease = 0;
  
  if(pattern == ignorePattern) return {history, weightDecrease, weightIncrease};
  
    for(let j = 0; j < weights[0][0].length; ++j) {
    // updating spark history
    history.in[j] = input[j] == 0 ? history.in[j] + 1 : 0;
    history.mid[neuron] = hidden[neuron] == 0 ? history.mid[neuron] + 1 : 0;

    const decrease = Settings.learningRate; 
    const increase = Settings.learningRate * 2 ** -history.in[j];

    // spike time depend plasticity
    const fired = hidden[neuron] == 1 || input[j] == 1;

    const i = neuron;
    
    if(history.mid[neuron] <= history.in[j]){
      weights[0][i][j] = fired && weights[0][i][j] > Settings.weightMinimum ? weights[0][i][j] - decrease : weights[0][i][j];
      weightDecrease += fired && weights[0][i][j] > Settings.weightMinimum ? 1 : 0;
    } else {
      weights[0][i][j] = fired && weights[0][i][j] < localMaximum ? weights[0][i][j] + increase : weights[0][i][j];
      weightIncrease += fired && weights[0][i][j] < Settings.weightMaximum ? 1 : 0;
    }
    
    weights[0][i][j] = weights[0][i][j] > localMaximum ? localMaximum : weights[0][i][j];
    weights[0][i][j] = weights[0][i][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[0][i][j];

  }

  return {history, weightDecrease, weightIncrease};
}

function trainItSparkLayer2(hidden, output, history, localMaximum){
  let weightDecrease = 0, weightIncrease = 0;
  for (let j = 0; j < weights[1][0].length; ++j) { 
    // updating spark history
    history.mid[j] = hidden[j] == 0 ? history.mid[j] + 1 : 0;
    history.out = output[0] == 0 ? history.out + 1 : 0;

    const decrease = Settings.learningRate; 
    const increase = Settings.learningRate * 2 ** -history.mid[j];

    // spike time depend plasticity
    const fired = output[0] == 1 || hidden[j] == 1;

    if(history.out <= history.mid[j]){
      weights[1][0][j] = fired && weights[1][0][j] > Settings.weightMinimum ? weights[1][0][j] - decrease : weights[1][0][j];
      weightDecrease += fired && weights[1][0][j] > Settings.weightMinimum ? 1 : 0;
    } else {
      weights[1][0][j] = fired && weights[1][0][j] < localMaximum ? weights[1][0][j] + increase : weights[1][0][j];
      weightIncrease += fired && weights[1][0][j] < Settings.weightMaximum ? 1 : 0;
    }
    weights[1][0][j] = weights[1][0][j] > localMaximum ? localMaximum : weights[1][0][j];
    weights[1][0][j] = weights[1][0][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[1][0][j];
  }
  return {history, weightDecrease, weightIncrease};
}

function trainItPerceptronLayer2(hidden, output, history, error){
  for (let j = 0; j < weights[1][0].length; ++j) { 
    // updating spark history
    history.mid[j] = hidden[j] == 0 ? history.mid[j] + 1 : 0;
    history.out = output[0] == 0 ? history.out + 1 : 0;

    weights[1][0][j] += Settings.learningRate * error * hidden[j];
    
    weights[1][0][j] = weights[1][0][j] > Settings.weightMaximum ? Settings.weightMaximum : weights[1][0][j];
    weights[1][0][j] = weights[1][0][j] < Settings.weightMinimum ? Settings.weightMinimum : weights[1][0][j];
  }
  return {history};
}

function trainPatch(patch, rightPatterns) {
  let weightIncrease = 0;
  let weightDecrease = 0;
  let sparks = 0;
  let iteration = 0;  
  let history = {in: [], mid: [], out: 0};

  // initializing spark history
  for(let one of patternToInput(0)) {
    history.in.push(1);
    history.mid.push(1);
  }

  let best = {f: 100, i: 0};
  for(let one of patch) {
    const input = patternToInput(one); 
    const hidden = layerCalc(input, weights, loads).mid;
    //console.log(`** ${JSON.stringify(hidden)}`);
    //const localMaximum = localMax(input);
    const localMaximum = Settings.weightMaximum; 

    let oldcheck = 100;
    //const result = trainItSparkLayer1(hack, hidden, history, localMaximum);
    let result = {history, weightDecrease, weightIncrease};
    for(let n = 0; n < weights[0].length; ++n){
      result = trainNeuron(one, hidden, result.history, localMaximum, n, rightPatterns[n%rightPatterns.length]);
    }
    //const newcheck = checkTrainingRandomly(512); 

    history = result.history;
    weightDecrease += result.weightDecrease;
    weightIncrease += result.weightIncrease;

    //oldcheck = newcheck.rights;
    //best.i = newcheck.falsePositives < best.f ? iteration : best.i;
    //best.f = newcheck.falsePositives < best.f ? newcheck.falsePositives : best.f;
    ++iteration;
  }
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease} cycles ${iteration} best ${best.f} at ${best.i}`);
  console.log(`history ${JSON.stringify(history)}`);
  console.log(`load ${loads[0][0]}`);
 
  for(let one of patch) {
    const input = patternToInput(one); 
    const all = layerCalc(input, weights, loads);

    // perceptron learning rule 
    const right = rightPatterns.includes(one) ? 1 : 0;
    const error = right - all.final[0];
    const result = trainItPerceptronLayer2(all.mid, all.final, history, error);
    
    history = result.history;
    ++iteration;
  }

  //weights[1][0] = [1,1,1, 1,1,1, 1,1,1];
  console.log(`weight increase ${weightIncrease} decrease ${weightDecrease} sparks ${sparks} cycles ${iteration} best ${best.f} at ${best.i}`);
  console.log(`history ${JSON.stringify(history)}`);
  console.log(`load ${loads[0][0]}`);

  return 0;
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
    const output = layerCalc(input, weights, loads).final;
    console.log(`${Math.floor(sample)} should be zero and is ${output[0]}`);
  }
}

function checkFullTraining(rightPatterns){
  let rights = 0;
  let falsePositive = 0;
  for(let i = 0; i < 2 ** 15; ++i){
    const input = patternToInput(i);
    const output = layerCalc(input, weights, loads).final;
    rights += rightPatterns.includes(i) && output[0] == 1 ? 1 : 0;
    falsePositive += output[0] == 1 ? 1 : 0;
  }
  //console.log(`rights ${rights/rightPatterns.length * 1e2} % falsePositives ${falsePositive/(2 ** 15) * 1e2} %`);
  return {rights: rights/rightPatterns.length * 1e2, falsePositives: falsePositive/(2 ** 15) * 1e2};
}

function checkLayerCalc(){
  let nothing = true;
  let sample = 0;
  for(let i=0; i < 2**15; ++i){
    //const result = layerCalc(patternToInput(i), weights, loads);
    const hack = calcLayer(patternToInput(i), weights[0], loads[0]);
    const result = {mid: hack, final: hack};
    let sumMid = 0;
    let sumFinal = 0;
    for (let one of result.mid){sumMid += one;}
    for (let one of result.final){sumFinal += one;}
    if ((sumMid || sumFinal) && sample < 9 ){
      console.log(`${JSON.stringify(result.mid)} ${JSON.stringify(result.final)}`);
      nothing = false;
      ++sample;
    }
  }
  if(nothing)console.log('nothing');
}

function cullHiddenLayer(rightPatterns, falseRate){
  for(let i = 0; i < weights[0].length; ++i){
    const result = checkNeuron(rightPatterns, weights[0][i], loads[0][i]);
    // if performance of neuron not acceptable its weights get set to zero or disconnected
    if (!(result.rights > 0 && result.falsePositive < falseRate)){
      console.log(`neuron ${i} culled rights ${result.rights} false ${result.falsePositive}`);
      for(let n = 0; n < weights[0][i].length; ++n) {
        weights[0][i][n] = 0;
      }
    }
  }
}

function checkNeuron(rightPatterns, Weights, Load){
  let rights = 0;
  let falsePositive = 0;
  for(let i = 0; i < 2 ** 15; ++i){
    const input = patternToInput(i);
    let out = calcSimpleNeuron(input, Weights, Load);
    rights += rightPatterns.includes(i) && out == 1 ? 1 : 0;
    falsePositive += out == 1 ? 1 : 0;
  }
  return {rights: rights/rightPatterns.length, falsePositive: falsePositive/(2**15)}
}

function analyzeSleep(data, rightPatterns){
  for(let time = 0; time < weights[0].length; ++time){
    let frequency = Array(rightPatterns.length).fill(0);
    for(let one of data){
      for(let c = 0; c < rightPatterns.length; ++c){
        frequency[c] += rightPatterns[c] == one ? 1 : 0; 
      }
    }
    console.log(`mostfrequent in timeslot ${time} : ${JSON.stringify(frequency)}`);
  }
}

function analyzeBlock(data, rightPatterns){
  for(let time = 0; time < weights[0].length; ++time){
    let frequency = Array(rightPatterns.length).fill(0);
    let block = 0;
    for(let one of data){
      for(let c = 0; c < rightPatterns.length; ++c){
        block = time % rightPatterns.length;
        frequency[c] += rightPatterns[c] == one && one != rightPatterns[block] ? 1 : 0; 
      }
    }
    console.log(`mostfrequent in timeslot ${time} : ${JSON.stringify(frequency)} , block ${block}`);
  }
}