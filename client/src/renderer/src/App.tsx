import { useState } from 'react'
import axios from 'axios'
import FolderPicker from './components/FolderPicker'
import SearchInterface from './components/SearchInterface'

function App(): JSX.Element {
  const [folderPath, setFolderPath] = useState<string>('')

  const indexImages = async (): Promise<void> => {
    try {
      const response = await axios.post('http://localhost:5000/index', {
        folder_path: folderPath,
        // Comment out to index all images in the folder
        image_count: 1000
      })

      console.log(response)
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div>
      <FolderPicker setFolderPath={setFolderPath} />
      <button onClick={indexImages}>Index Images</button>
      <SearchInterface />
    </div>
  )
}

export default App
