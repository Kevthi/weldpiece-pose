import React from 'react'
import './Workspace.css'

const Workspace = ({ children }: any) => {
    return (
        <div className="Workspace">
            {children}
        </div>
    )
}

export default Workspace