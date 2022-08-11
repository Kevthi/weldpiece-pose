import React, { useEffect, useRef } from "react";
import { WindowManager } from "./WindowManager";
import { useState } from "react";

const ThreeWin = ({ scene }) => {
    const dispatch = useDispatch();
    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    let timer;

    const resize = (sceneManager, container) => {
        clearTimeout(timer)

        timer = setTimeout(() => {
            let width = container.clientWidth;
            let height = container.clientHeight;
            sceneManager.resizeCanvas(width, height)
        }, 4)
    }

    const onMouseDown = (event, canvas) => {
        const rect = canvas.getBoundingClientRect();
        const mousePos = [event.clientX - rect.left, event.clientY - rect.top];
        dispatch(setMouseDownPos(mousePos));

    }
    const onMouseUp = (event, canvas) => {
        const rect = canvas.getBoundingClientRect();
        const mouseUpPos = [event.clientX - rect.left, event.clientY - rect.top];
        const mouseDownPos = store.getState().threeCanvasClick.mouseDownPos
        if (mouseDownPos[0] === mouseUpPos[0] && mouseDownPos[1] === mouseUpPos[1]) {
            const mouse_x = (mouseUpPos[0] / canvas.clientWidth) * 2 - 1;
            const mouse_y = -(mouseUpPos[1] / canvas.clientHeight) * 2 + 1;
            const clicked = [mouse_x, mouse_y];
        }
    }



    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;

        canvas.addEventListener('mousedown', (e) => onMouseDown(e, canvas))
        canvas.addEventListener('mouseup', (e) => onMouseUp(e, canvas))

        let windowManager = new WindowManager(sceneManager, canvas, container);
        let resizeObs = new ResizeObserver(() => { resize(windowManager, containerRef.current) })
        resizeObs.observe(containerRef.current);
        const animate = () => {
            windowManager.update();
            requestAnimationFrame(animate);
        }

        animate(sceneManager);
        return () => {
            windowManager.dispose();
            resizeObs.disconnect();
            // remove event listeners
            canvas.removeEventListener('mousedown', onMouseDown)
            canvas.removeEventListener('mouseup', onMouseUp)
        }

    })
    return (
        <div ref={containerRef} className="threeContainer">
            <canvas ref={canvasRef} className="threeCanvas"></canvas>
        </div>
    )
}

export default ThreeWin
