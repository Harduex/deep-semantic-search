import { useRef, useState } from 'react'
import axios from 'axios'
import styles from './SearchInterface.module.scss'

const SearchInterface: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [searchResults, setSearchResults] = useState<[string, number][]>([])
  const imageContainerRef = useRef<HTMLDivElement>(null)

  const handleSearch = async (): Promise<void> => {
    const response: { data: Record<string, number> } = await axios.post(
      'http://localhost:5000/search-text',
      {
        text: searchTerm,
        number_of_images: 100
      }
    )
    const sortedResults = Object.entries(response.data).sort((a, b) => b[1] - a[1])
    setSearchResults(sortedResults)
    imageContainerRef.current?.scrollTo(0, 0)
  }

  return (
    <div className={styles.container}>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className={styles.input}
      />
      <button onClick={handleSearch} className={styles.button}>
        Search
      </button>
      <div className={styles.imageContainer} ref={imageContainerRef}>
        {searchResults.map(([imagePath, score]) => (
          <div key={imagePath} className={styles.imageWrapper}>
            <img src={`file://${imagePath}`} className={styles.image} />
            <p>{score}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SearchInterface
