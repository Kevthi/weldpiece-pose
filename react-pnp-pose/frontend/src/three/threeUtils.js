import * as THREE from 'three';


export const listsToMatrix4 = (lists) => {
  let matrix = new THREE.Matrix4();
  matrix.set(
    lists[0][0], lists[0][1], lists[0][2], lists[0][3],
    lists[1][0], lists[1][1], lists[1][2], lists[1][3],
    lists[2][0], lists[2][1], lists[2][2], lists[2][3],
    lists[3][0], lists[3][1], lists[3][2], lists[3][3]
  );
  return matrix;
}

