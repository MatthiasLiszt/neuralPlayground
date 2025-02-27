// getting 3x3 patterns from traininsData

function getPatternsFromData(data){
  const store = [];
  const patterns = [];
  for (let one of data) {
    //first three lines
    const k1 = one.pattern % (2 ** 9);

    //the three lines in the middle
    const k2 = Math.floor(one.pattern / 8) % (2 ** 9);

    //the last three lines
    const k3 = Math.floor(one.pattern / 64) % (2 ** 9);

    store[k1.toString()] = k1;
    store[k2.toString()] = k2;
    store[k3.toString()] = k3;
  }

  for (let one of store) {
    if (one !== undefined) patterns.push({p: one});
  }
  return patterns;
}

// groups into K groups and returns the data sets, mean distance etc. 
function groupByK(data, k) {
  const centroids = [];
  for(let i = 0; i < k; ++i) {
    const centroid = Math.floor(Math.random() * data.length);
    centroids.push({position: data[centroid].p, distance: 1});
  }
  
  let grouped = []
  
  for (let one of centroids){
    grouped.push([])
  }

  for(let one of data){
    let oldDistance = 9;
    let bestgroup = 0;
    for (let i = 0; i < k; ++i){
      newDistance = calcDistance(one.p, centroids[i].position);
      if(newDistance < oldDistance) {
        bestgroup = i;
        oldDistance = newDistance;
      }
    }
    grouped[bestgroup].push(one.p);
    // worst distance for current centroid
    centroids[bestgroup].distance = oldDistance > centroids[bestgroup].distance ? oldDistance : centroids[bestgroup].distance;
    // looking for better centroid
    centroids[bestgroup] = newCentroid(grouped[bestgroup], centroids[bestgroup]);
    console.log(`pattern ${one.p} centroid ${centroids[bestgroup].position}`);
  }
  
  return {grouped, centroids}
}

// calculates distance(error) from centroid
function calcDistance(pattern, centroid){
  let errors = 0;
  for(let i = 0; i < 9; ++i){
    errors = (centroid & (2 ** i)) != (pattern & (2 ** i)) ? errors + 1 : errors;
  }
  return errors;
}

function newCentroid(data, oldCentroid) {
  let bestDistance = oldCentroid.distance;
  let bestCentroid = oldCentroid.position;
  for (let centroid of data) {
    let worstDistance = 0;
    for (let compare of data) {
      if (centroid != compare) {
        let distance = calcDistance(compare, centroid);
        worstDistance = distance > worstDistance ? distance : worstDistance;
      }
    }
    if (bestDistance > worstDistance) {
      bestDistance = worstDistance;
      bestCentroid = centroid;
    }
  }
  return {position: bestCentroid, distance: bestDistance};
}

function logDig(n){
  return Math.log(n) / Math.log(2);
}