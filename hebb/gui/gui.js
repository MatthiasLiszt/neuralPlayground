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