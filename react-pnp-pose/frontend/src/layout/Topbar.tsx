import React from 'react'
import './Topbar.css'



const Topbar = ({ children }: any) => {
    return (
        <div className="Topbar">
            <div className="TopbarTabRow">
                {children}
            </div>
            <div className="TopbarRowBelow"></div>
        </div>
    )
}

export default Topbar