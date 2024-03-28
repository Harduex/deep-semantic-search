import './App.css'
import { API_URL, OLLAMA_URL } from './constants'

function App() {
  return (
    <div>
    <button onClick={() => fetch(API_URL).then(res => res.json()).then(data => console.log(data))}>Fetch Data</button>   
    <button onClick={() => fetch(OLLAMA_URL).then(res => res.text()).then(data => console.log(data))}>Ollama</button>   
    </div>
  )
}

export default App
