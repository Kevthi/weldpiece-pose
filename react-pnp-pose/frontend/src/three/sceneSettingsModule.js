import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { store, subscribe } from '../app/store'

export class SceneSettingsModule {
  constructor(scene) {
    this.scene = scene
    this.setBackgroundColor()
  }
  setBackgroundColor() {
    subscribe('settings.sceneBackgroundColor', (state) => {
      console.log("setBackgroundColor", state.settings.backgroundColor)
      let sceneBackgroundColor = state.settings.sceneBackgroundColor
      // check if valid hex color
      if (sceneBackgroundColor.match(/^#[0-9A-F]{6}$/i)) {
        this.scene.background = new THREE.Color(sceneBackgroundColor)
      }
    })
  }
}


