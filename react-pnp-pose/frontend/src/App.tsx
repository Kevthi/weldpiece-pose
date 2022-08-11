import './App.css'
import Topbar from './layout/Topbar'
import InitPosePage from './pages/init-pose/InitPosePage'

function App() {
  const on_btn_click = () => {
    console.log("btn clicked")
  }

  let window = 1

  return (
    <div className="App">
      <Topbar />
      <InitPosePage />




    </div>
  )
}

export default App
