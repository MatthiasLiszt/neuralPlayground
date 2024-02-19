
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

function store(id) {
  patternToSave.sign = id;
  var Id = document.getElementById('b' + id);
  Id.style.backgroundColor = "green";
}

function saveToDisk() {
  for (var one of DATA) {
    if (one.pattern === patternToSave.pattern) {
      message('error: pattern is already there');
      return;
    }
  }
  DATA.push(patternToSave);
  localStorage.setItem(storeName, JSON.stringify(DATA));
  message('new data saved to disk');
  loadData();
}

function message(text) {
  var Id = document.getElementById('response');
  Id.textContent = text;
}

function save() {
  var pattern = 0;
  var value = 1;
  for(var i = 0; i < DATAINPUT.length; ++i) {
    pattern += DATAINPUT[i] ? value : 0;
    value *= 2;
  }
  patternToSave.pattern = pattern;
  if (patternToSave.sign !== undefined) {
    saveToDisk();
    resetDataInput();
    patternToSave.sign = undefined;
    patternToSave.pattern = 0;
  } else {
    message('could not be saved');
  }
}

function loadData() {
  var Id = document.getElementById('dataCount');
  var content = localStorage.getItem(storeName);
  if (content === null) {
    Id.textContent = '0 entries';
  } else {
    DATA = JSON.parse(content);
    Id.textContent = DATA.length + ' entries';
  }
}

