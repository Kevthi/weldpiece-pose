// import THREE as es6 module
import * as THREE from 'three';
import { store, subscribe } from '../app/store';



export class ClickHandler {
  constructor(scene, camera, pc_module) {
    console.log("ClickHandler")
    this.scene = scene
    this.camera = camera
    this.raycaster = new THREE.Raycaster()
    this.raycaster.params.Points.threshold = 0.001;
    this.clickAble = []
    this.selected = null
    this.pc_module = pc_module
    this.listenToClick()
  }

  listenToClick() {
    console.log("listenToClick")
    subscribe("threeCanvasClick.clickedPos", (state) => {
      let clickedPos = state.threeCanvasClick.clickedPos
      const mouse = new THREE.Vector2();
      mouse.x = clickedPos[0]
      mouse.y = clickedPos[1]
      this.raycaster.setFromCamera(mouse, this.camera);
      const intersections = this.raycaster.intersectObjects(this.pc_module.scans, false);
      const intersection = (intersections.length) > 0 ? intersections[0] : null;
      console.log(intersection)
      if (intersection) {
        this.createSphereAtClick(intersection.point)
        let point = intersection.point
        let index = intersection.index
        let pos = [point.x, point.y, point.z]
        let dictKey = index + this.pc_module.getDictIdx(pos[0], pos[1], pos[2])
        console.log(dictKey)
        let info = this.pc_module.pointHashTable[dictKey]
        console.log("index", info[0])
        console.log("pointcloud", info[1])
      }


    })
  }

  attachEventListener() {

    // get mouse position
    let mouse = new THREE.Vector2();
    let rect = that.canvas.getBoundingClientRect()
    //console.log(rect.x, rect.y)
    //console.log(rect)
    mouse.x = ((event.clientX - rect.x) / this.canvas.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.y) / this.canvas.clientHeight) * 2 + 1;
    console.log(mouse)
    this.raycaster.setFromCamera(mouse, this.camera);
    const intersections = this.raycaster.intersectObjects(this.clickAble, false);
    console.log(intersections)
    const intersection = (intersections.length) > 0 ? intersections[0] : null;
    console.log(intersection)
    // if interscetion is not null
    if (intersection) {
      this.createSphereAtClick(intersection.point)


    }
  }

  appendClickable(obj) {
    this.clickAble.push(obj)
  }

  createSphereAtClick(pos) {
    this.removeSphere()
    let geometry = new THREE.SphereGeometry(0.002, 32, 32);
    // semi transparent sphere
    let material = new THREE.MeshBasicMaterial({
      color: 0xffff00,
      transparent: true,
      opacity: 0.8
    });
    let sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(pos.x, pos.y, pos.z);
    this.selected = sphere
    this.scene.add(sphere);
  }

  removeSphere() {
    if (this.selected) {
      this.scene.remove(this.selected)
      this.selected = null
    }
  }

}
