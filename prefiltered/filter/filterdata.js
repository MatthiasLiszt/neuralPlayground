const DOT = [
  {pattern: 0 ,result: false},
  {pattern: 0b1 ,result: true},
  {pattern: 0b10 ,result: true},
  {pattern: 0b11 ,result: false},
  {pattern: 0b100 ,result: true},
  {pattern: 0b101 ,result: false},
  {pattern: 0b110 ,result: false},
  {pattern: 0b111 ,result: false},
  {pattern: 0b1000 ,result: true},
  {pattern: 0b1001 ,result: false},
  {pattern: 0b1010 ,result: false},
  {pattern: 0b1011 ,result: false},
  {pattern: 0b1100 ,result: false},
  {pattern: 0b1101 ,result: false},
  {pattern: 0b1110 ,result: false},
  {pattern: 0b1111 ,result: false}
];

const HORIZONTAL = [
  {pattern: 0 ,result: false},
  {pattern: 0b1 ,result: false},
  {pattern: 0b10 ,result: false},
  {pattern: 0b11 ,result: true},
  {pattern: 0b100 ,result: false},
  {pattern: 0b101 ,result: false},
  {pattern: 0b110 ,result: false},
  {pattern: 0b111 ,result: false},
  {pattern: 0b1000 ,result: false},
  {pattern: 0b1001 ,result: false},
  {pattern: 0b1010 ,result: false},
  {pattern: 0b1011 ,result: false},
  {pattern: 0b1100 ,result: true},
  {pattern: 0b1101 ,result: false},
  {pattern: 0b1110 ,result: false},
  {pattern: 0b1111 ,result: false}
];

const VERTICAL = [
  {pattern: 0 ,result: false},
  {pattern: 0b1 ,result: false},
  {pattern: 0b10 ,result: false},
  {pattern: 0b11 ,result: false},
  {pattern: 0b100 ,result: false},
  {pattern: 0b101 ,result: true},
  {pattern: 0b110 ,result: false},
  {pattern: 0b111 ,result: false},
  {pattern: 0b1000 ,result: false},
  {pattern: 0b1001 ,result: false},
  {pattern: 0b1010 ,result: true},
  {pattern: 0b1011 ,result: false},
  {pattern: 0b1100 ,result: false},
  {pattern: 0b1101 ,result: false},
  {pattern: 0b1110 ,result: false},
  {pattern: 0b1111 ,result: false}
];

const DIAGONAL = [
  {pattern: 0 ,result: false},
  {pattern: 0b1 ,result: false},
  {pattern: 0b10 ,result: false},
  {pattern: 0b11 ,result: false},
  {pattern: 0b100 ,result: false},
  {pattern: 0b101 ,result: false},
  {pattern: 0b110 ,result: true},
  {pattern: 0b111 ,result: false},
  {pattern: 0b1000 ,result: false},
  {pattern: 0b1001 ,result: true},
  {pattern: 0b1010 ,result: false},
  {pattern: 0b1011 ,result: false},
  {pattern: 0b1100 ,result: false},
  {pattern: 0b1101 ,result: false},
  {pattern: 0b1110 ,result: false},
  {pattern: 0b1111 ,result: false}
];

const CORNER = [
  {pattern: 0 ,result: false},
  {pattern: 0b1 ,result: false},
  {pattern: 0b10 ,result: false},
  {pattern: 0b11 ,result: false},
  {pattern: 0b100 ,result: false},
  {pattern: 0b101 ,result: false},
  {pattern: 0b110 ,result: false},
  {pattern: 0b111 ,result: true},
  {pattern: 0b1000 ,result: false},
  {pattern: 0b1001 ,result: false},
  {pattern: 0b1010 ,result: false},
  {pattern: 0b1011 ,result: true},
  {pattern: 0b1100 ,result: false},
  {pattern: 0b1101 ,result: true},
  {pattern: 0b1110 ,result: true},
  {pattern: 0b1111 ,result: false}
];

const VERTICALLEFT = [
  {pattern: 0 ,result: false},
  {pattern: 0b1 ,result: false},
  {pattern: 0b10 ,result: false},
  {pattern: 0b11 ,result: false},
  {pattern: 0b100 ,result: false},
  {pattern: 0b101 ,result: false},
  {pattern: 0b110 ,result: false},
  {pattern: 0b111 ,result: false},
  {pattern: 0b1000 ,result: false},
  {pattern: 0b1001 ,result: false},
  {pattern: 0b1010 ,result: true},
  {pattern: 0b1011 ,result: false},
  {pattern: 0b1100 ,result: false},
  {pattern: 0b1101 ,result: false},
  {pattern: 0b1110 ,result: false},
  {pattern: 0b1111 ,result: false}
];