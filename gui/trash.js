// only supports sigmoid function as activation function
function backprop4Layer(outputs, Weights, Bias, target, input) {
  var Changed = {w: [] , b: []};

  //create targets from target value
  var targets = [];
  for (var i = 0; i < 10; ++i) {
    var value = 0.001;
    value = i == target ? 1 : value;
    targets.push(value);
  }

  // false value for the relevant dj 
  var Fj = 1;

  // first layer
  // comes before because we need the unchanged weights of the last layer
  for (var i = 0; i < Settings.neurons[1]; ++i) {
    // short method
    var sum = 0;
    for (var k = 0; k < Settings.neurons[2]; ++k) {
      var ol = outputs.O[1][k];
      var dl = ol * (1 - ol);
      sum += ol * Weights[1][k][i];
    }
    var oj = outputs.O[0][i];
    var dj = oj * (1 - oj) * sum;

    for (var j = 0; j < Settings.neurons[0]; ++j) {
      var oi = input[j];
      Weights[0][i][j] += - Settings.learningRate * oi * dj;
    }
  }

  // second layer
  // comes before because we need the unchanged weights of the last layer
  var SecondLE = [];
  for (var i = 0; i < Settings.neurons[2]; ++i) {
    // short method
    var sum = 0;
    for (var k = 0; k < Settings.neurons[3]; ++k) {
      var ol = outputs.last[k];
      var dl = ol * (1 - ol);
      sum += dl * Weights[2][k][i];
    }
    SecondLE.push(sum);
    var oj = outputs.O[1][i];
    var dj = oj * (1 - oj) * sum;

    for (var j = 0; j < Settings.neurons[1]; ++j) {
      var oi = outputs.O[0][j];
      Weights[1][i][j] += - Settings.learningRate * oi * dj;
    }
  }
  console.log('2nd layer sums ' + SecondLE.join('  '));
  // last layer
  for (var i = 0; i < Settings.neurons[3]; ++i) {
    // short method 
    var oj = outputs.last[i];
    var tj = targets[i];
    var dj = (oj - tj) * oj * (1 - oj);
    // dE/dwj
    //var dEIdwj = oj * dj;

    // long method
    var dEtIdoj = oj - tj;
    var dojIdNetj = oj * (1 - oj);
    // var dj = dEtIdoj * dojIdNetj;

    for (var j = 0; j < Settings.neurons[2]; ++j) {
      var oi = outputs.O[1][j];
      Weights[2][i][j] += - Settings.learningRate * oi * dj;
    }
  }

  Changed.w = Weights;
  Changed.b = Bias;
  return Changed;
}