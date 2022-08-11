import React from 'react'
import './PageWin.css'

const PageWin = ({ children }: any) => {
    return (
        <div className="PageWin">
            {children}
        </div>
    )
}

export default PageWin