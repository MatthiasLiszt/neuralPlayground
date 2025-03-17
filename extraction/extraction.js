const sigmoid = x => 1/(1 + Math.E**(-x));

// extraction of filters from an existing solution 

function getLowestValue(data){
  let lowest = Math.abs(data[0]);
  for (let one of data) {
    lowest = Math.abs(one) < Math.abs(lowest) ? one : lowest;
  }
  return lowest;
}

function getConnectionsAboveLimit(data, limit) {
  let connections = [];
  for (let one of data) {
    let count = 0;
    for (let weight of one) {
      count = weight > limit ? ++count : count;
    }
    connections.push(count);
  }
  return connections;
}

function numberToInput(pattern, max) {
  var rest = pattern;
  var input = [];
  while(rest > 0) {
    input.push(rest % 2);
    rest *= 0.5;
    rest = Math.floor(rest);
  }
  while(input.length < max) {
    input.push(0);
  }
  return input;
}

function calculate(input, weights, bias) {
  var sum = 0;
  for (var j = 0; j < input.length; ++j) {
    sum += input[j] * weights[j] + bias;
  }
  return sigmoid(sum);
}