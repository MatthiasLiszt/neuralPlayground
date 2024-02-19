var identity = (x) => x;
var sigmoid = (x) => 1/(1 + Math.E ** (-x));

var lrate = 0.01; // learning rate
var target = 0.5;

// Layer I and Layer K use identity function
// Layer J uses sigmoid function

//Layer I 
var I = { net: [1, 1], out: []};

// weights of Layer J
var Jw = [[0.5, 0.5], [0.5, 0.5]];

// weights of Layer K 
var Kw = [[0.5, 0.5]];


// forward pass
console.log('forward pass');

// calculation I.out
for( let one of I.net) {
  I.out.push(identity(one));
}

console.log('I ' + JSON.stringify(I));

// calculating for J 
var J = {net: [], out: [] };

var i = 0;
for(let ws of Jw) {
  var sum = 0;
  for (let w of ws){
    sum += w * I.out[i];
  }
  ++i;
  J.net.push(sum);
  J.out.push(sigmoid(sum));
}

console.log('J ' + JSON.stringify(J));

// calculating for K
var K = {net: [], out: [] };

i = 0;
for(let ws of Kw) {
  var sum = 0;
  for (let w of ws){
    sum += w * J.out[i];
  }
  ++i;
  K.net.push(sum);
  K.out.push(identity(sum));
}

console.log('K ' + JSON.stringify(K));

// backward pass
console.log('backward pass');

//calculating gradients (all start with V)
var dk0 = K.out[0] - target;
var Vwj0 = dk0 * J.out[0];

var JSum = dk0 * Kw[0][0];
var dj0 = J.out[0] * (1 - J.out[0]) * JSum;
var Vwi0 = dj0 * I.out[0];

console.log(`dk0 ${dk0} dj0 ${dj0}`);
console.log(`Vwj0 ${Vwj0} Vwi0 ${Vwi0}`);
//adjusting weights
Jw[0][0] -= lrate * Vwi0;
Kw[0][0] -= lrate * Vwj0;

console.log(`weights ${Jw[0][0]} ${Kw[0][0]}`);
