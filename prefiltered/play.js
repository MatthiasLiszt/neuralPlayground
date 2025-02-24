//settings
const Settings = {
  layers: 2,
  neurons: [30, 10],
  learningRate: 0.01,
  activation: 'stepfunction',
  f: x => x >= 1 ? 1 : 0
}

// filter
const xor = (a, b) => (a && !b) || (b && !a); 
const verticalFilter = (q) => xor(q[0] && q[2] , q[1] && q[3]);
const horizontalFilter = (q) => xor(q[0] && q[1] , q[2] && q[3]);
const diagonalFilter = (q) => xor(q[0] && q[3], q[1] && q[2]);
const dotFilter = (q) => (q[0] + q[1] + q[2] + q[3]) == 1;
const cornerFilter = (q) => (q[0] + q[1] + q[2] + q[3]) == 3;
const allFilters = [verticalFilter, horizontalFilter, diagonalFilter, dotFilter, cornerFilter];

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

// gui and other things

var Response = document.getElementById('response');

function checkIt() {
  var o = layerCalc(preFiltering(DATAINPUT), weights, bias);
  Response.textContent = `it's a ${findMax(o).at}`;
  console.log(JSON.stringify(o));
}

var DATAINPUT = new Array(15).fill(false);
var patternToSave = {pattern: 0, sign: undefined};
var storeName = 'neuralnetdata';
var DATA = [];

function change(id) {
  var Id = document.getElementById('d' + id);
  DATAINPUT[id] = !DATAINPUT[id];
  if(DATAINPUT[id]) {
    Id.style.backgroundColor = "green";
  } else {
    Id.style.backgroundColor = "grey";
  }
}

function resetDataInput() {
  DATAINPUT = DATAINPUT.map( x => x = false);
  for(var i = 0; i < DATAINPUT.length; ++i) {
    var Id = document.getElementById('d' + i);
    Id.style.backgroundColor = "grey";
  }
  for(var i = 0; i < 10; ++i) {
    var Id = document.getElementById('b' + i);
    Id.style.backgroundColor = "lightgrey";
  }
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

function preFiltering(pattern) {
  const i = [];
  
  for (let one of pattern) {
    i.push(one ? 1 : 0);
  }
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
  console.log(JSON.stringify(preFiltered));
  return preFiltered;
}
