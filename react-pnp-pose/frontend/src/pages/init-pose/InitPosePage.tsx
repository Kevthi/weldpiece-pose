import React from 'react'
import './InitPosePage.css'
import PageWin from '../../layout/PageWin'
import InitPoseSidebar from './components/InitPoseSidebar'
import Workspace from '../../layout/Workspace'
import { Canvas, useThree } from '@react-three/fiber'
import * as THREE from 'three';
import { useEffect } from 'react'
import { ArcballControls, PerspectiveCamera, MapControls } from "@react-three/drei"


const InitPosePage = () => {
    //<ArcballControls enablePan={false} dampingFactor={1000} wMax={100.0} setGizmosVisible={true} />
    return (
        <PageWin>
            <InitPoseSidebar />
            <Workspace>
                <Canvas>

                    <ambientLight intensity={0.1} />
                    <directionalLight color="red" position={[0, 0, 3]} />
                    <MapControls />
                    <mesh>
                        <boxGeometry />
                        <meshStandardMaterial />
                    </mesh>


                </Canvas>

            </Workspace>


        </PageWin>
    )
}

export default InitPosePage