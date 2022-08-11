import React from 'react'
import './Sidebar.css'

const Sidebar = ({ children }: any) => {
    return (
        <div className="Sidebar">
            {children}
        </div>
    )
}

export default Sidebar