import { ChangeEvent, FC, useEffect, useRef } from 'react'

interface FolderPickerProps {
  setFolderPath: (path: string) => void
}

const FolderPicker: FC<FolderPickerProps> = ({ setFolderPath }) => {
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.setAttribute('webkitdirectory', '')
    }
  }, [])

  const handleFolderChange = (e: ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files?.length) {
      const filePath = e.target.files[0].path
      const folderPath = filePath.substring(0, filePath.lastIndexOf('/'))
      setFolderPath(folderPath)
    }
  }

  return (
    <div>
      <input type="file" ref={inputRef} onChange={handleFolderChange} />
    </div>
  )
}

export default FolderPicker
