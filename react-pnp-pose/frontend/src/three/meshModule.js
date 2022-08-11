import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { store, subscribe } from '../app/store'

export class MeshModule {
  constructor(scene) {
    this.scene = scene

    this.loadObjOnProjectSelect()
    this.showHideMesh()
    this.activeMesh = null
    this.setOpacity()
  }

  setOpacity() {
    subscribe('settings.meshOpacity', (state) => {
      console.log("setOpacity", state.settings.meshOpacity)
      let meshOpacity = state.settings.meshOpacity
      if (this.activeMesh == null) return
      let object = this.activeMesh
      console.log("setting opacity", meshOpacity)
      object.traverse(function (child) {
        if (child instanceof THREE.Mesh) {
          child.material.transparent = true
          child.material.opacity = meshOpacity / 100.0
        }
      });
    })
  }

  loadObjOnProjectSelect() {

    subscribe('selectedProject.projectName', (state) => {
      if (this.activeMesh != null) {
        this.scene.remove(this.activeMesh)
      }
      let projectName = state.selectedProject.projectName
      console.log("loadObjOnProjectSelect")
      const url = "http://127.0.0.1:5000/get-mesh?project=" + projectName
      const loader = new OBJLoader();
      loader.load(url, (obj) => {
        // create phong material
        const material = new THREE.MeshPhongMaterial({
          color: 0xaaaaaa,
          specular: 0xccccaa,
          shininess: 2,
          flatShading: true
        });
        // add material to object
        obj.traverse(function (child) {
          if (child instanceof THREE.Mesh) {
            child.material = material
          }
        })
        this.activeMesh = obj
        this.scene.add(obj)
      }
      )
    })
  }

  showHideMesh() {
    //this.pubsub.subscribe('show_hide_mesh', (showTrue) => {
    subscribe('selectedProject.showMesh', (state) => {
      let showMesh = state.selectedProject.showMesh
      if (this.activeMesh == null) return
      let object = this.activeMesh
      object.traverse(function (child) {
        if (child instanceof THREE.Mesh) {
          child.visible = showMesh;
        }
      });


    })
  }


}
