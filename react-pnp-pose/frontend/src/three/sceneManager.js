import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { store, subscribe } from '../app/store'
//import { ClickHandler } from './clickHandler';
import { MeshModule } from './meshModule'
import { PointcloudModule } from './pointcloudModule';
import { CameraLaserModule } from './cameraLaserModule';
import { SceneSettingsModule } from './sceneSettingsModule';
import { ClickHandler } from './clickHandler';


export class SceneManager {
  constructor() {

    this.scene = this.createScene()
    this.initLights()
    this.camera = this.createCamera()
    this.initSceneModules(this.scene)

  }

  initSceneModules(scene) {
    this.meshModule = new MeshModule(scene)
    this.pointcloudModule = new PointcloudModule(scene)
    this.cameraLaserModule = new CameraLaserModule(scene)
    this.sceneSettingsModule = new SceneSettingsModule(scene)
    this.clickHandler = new ClickHandler(scene, this.camera, this.pointcloudModule)
  }

  getScene() {
    return this.scene
  }

  initLights() {
    // add 3 point lights to scene
    const light = new THREE.PointLight(0xffffff, 1, 30);
    light.position.set(5, 5, 5);
    this.scene.add(light);
    const light2 = new THREE.PointLight(0xffffff, 1, 30);
    light2.position.set(5, -5, 5);
    this.scene.add(light2);

  }

  createCamera() {
    //let cameraPos = store.getState().selectedProject.cameraPos
    let cameraPos = [5, 5, 5]
    var camera = new THREE.PerspectiveCamera(45, 1 / 1, 0.1, 1000)
    this.scene.add(camera)
    camera.up.set(0, 0, 1)
    camera.position.set(cameraPos[0], cameraPos[1], cameraPos[2])
    camera.lookAt(this.scene.position)
    return camera
  }

  getCamera() {
    return this.camera
  }



  createScene() {
    var scene = new THREE.Scene()
    //add axeshelper
    var axesHelper = new THREE.AxesHelper(1);
    scene.add(axesHelper);

    return scene
  }

}