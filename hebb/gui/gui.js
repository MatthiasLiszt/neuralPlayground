// gui functions

function turnDRAWonOff(){
  DRAW = !DRAW;
  console.log(`drawing is ${DRAW ? 'on' : 'off'}`);
}

function drawPixel(event){
  const ctx = Canvas.getContext("2d");
  let rect = Canvas.getBoundingClientRect();
  let x = event.clientX - rect.left;
  let y = event.clientY - rect.top;
  if(DRAW){
    let xy = fillGrid(x,y);
    let X = xy.X * PIXELSIZE;
    let Y = xy.Y * PIXELSIZE;
    let O = PIXELSIZE;
    //console.log(`color ${xy.color} x ${X} y ${Y}`);
    ctx.fillStyle = xy.color;
    ctx.fillRect(X, Y, O, O);
  }
}

function fillGrid(x,y){
  const X = Math.floor(x / PIXELSIZE);
  const Y = Math.floor(y / PIXELSIZE);
  //console.log(`x ${X} y ${Y}`);
  GRID[X][Y] = GRID[X][Y] == 0 ? 1 : 0;
  return {X, Y, color: GRID[X][Y] == 1 ? 'black' : 'white'}; 
}

function addPattern(){
  PATTERNS.push(GRID);
  const ctx = Canvas.getContext("2d");
  ctx.clearRect(0,0, Canvas.width, Canvas.height);
  const shrink = 0.25;
  Canvas.width *= shrink;
  Canvas.height *= shrink;
  const O = PIXELSIZE * shrink;
  ctx.fillStyle = 'blue';
  for (let x = 0; x < GRID.length; ++x) {
    for (let y = 0; y < GRID.length; ++y) {
      if(GRID[x][y] == 1){ctx.fillRect(x * O, y * O, O, O);}
    }
  }
  let image = new Image();
  image.src = Canvas.toDataURL();
  IMAGES.push(PatternsToLearn.appendChild(image));
  Canvas.width *= (1/shrink);
  Canvas.height *= (1/shrink);
  initGrid();
}

function removePattern(){
  PATTERNS.pop();
  PatternsToLearn.removeChild(IMAGES.pop());
}

function initGrid(){
  GRID = [];
  for(let i = 0; i < (Canvas.width/PIXELSIZE); ++i){
    let line = Array(Canvas.width / PIXELSIZE).fill(0);
    GRID.push(line);
  }
}

function trainPattern(){
  const X = Math.floor(Canvas.width/PIXELSIZE);
  const N = X * X;
  Settings.neurons = [N, PATTERNS.length * 3, 1];
  if(PATTERNS.length > 0){
    WorkingLight.style.background = "red";
    const traindata = generateRandomTrainingData(0.3, 1e4, PATTERNS);
    weights = initWeights(0.25, 0.1);
    switch(LearningMethod.value){
      case "competitiveOnly":
        let rate = 1;
        let repeat = 0;
        do{ 
          rate = trainPatchCompetitiveOnly(traindata);
          ++repeat;
        } while(repeat < 23 && rate > 0.9);
        FeedBack.textContent = rate > 0.9 ? "training failed" : "training successful"; 
        break;
      case "competitivePerceptron":
        trainPatchCompetitivePerceptron(traindata, PATTERNS);
        break;
      case "perceptronOnly":
        Settings.neurons[1] = PATTERNS.length * 4;
        weights = initWeights(0.25, 0.1);
        trainPatchPerceptronOnly(traindata, PATTERNS, createDeficiencies(0.9));
        break;
    }
    WorkingLight.style.background = "green";
  }
  else {
    console.log('no patterns to learn');
  }
}

function testPattern(){
  ResultLight.style.background = "grey";
  const result = layerCalc(GRID.flat(), weights).final;
  console.log(JSON.stringify(result));
  ResultLight.style.background = result[0] == 1 ? 'green' : 'grey';
}

