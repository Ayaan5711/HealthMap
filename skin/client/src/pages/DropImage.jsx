import React, { useRef, useState } from 'react'
import uploadImg from '../assets/images/cloud-upload-regular-240.png'
import { toast } from 'react-toastify'
import Wrapper from '../assets/wrappers/DropImage'
import axios from 'axios'

const DropImage = ({ onFileChange, setShowBox, setIsLoading, setContext }) => {
    const wrapperRef = useRef(null)

    const [image, setImage] = useState(null);

    const oDragEnter = () => wrapperRef.current.classList.add('dragover');

    const oDragLeave = () => wrapperRef.current.classList.remove('dragover');

    const onDrop = () => wrapperRef.current.classList.remove('dragover');

    const onImageDrop = (e) => {
        const newImg = e.target.files[0];
        const type = newImg.type
        if (type === "image/jpeg" || type === "image/png") {
            setImage(newImg);
            onFileChange(newImg)
        }
        else {
            toast.error('Please Upload Only Image (jpg/png)')
        }
    }

    const imageRemove = () => {
        setImage(null);
    }
    
    const handleClick = async () => {
        if (image) {
            try {
                // Create FormData to send image to backend
                const formData = new FormData();
                formData.append('image', image);
                
                // // Send image to backend
                await axios.post("http://localhost:5000/prediction", formData)
                .then((res) => {
                    // console.log(res.data);
                    setContext(res.data)
                })
                .catch((err) => {
                    console.log(err);
                })

                toast.success('Image uploaded successfully!')           
                setIsLoading(true)
                setTimeout(() => {
                    setIsLoading(false)
                },1500)
                
                setShowBox(false)
                
            } catch (err) {
                toast.error('Error in uploading image!')           
                console.error('Error uploading image:', err);
            }
        }
    }
    
    return (
        <Wrapper>
            {!image &&
                (
                    <div
                        ref={wrapperRef}
                        className='drop-file-input'
                        onDragEnter={oDragEnter}
                        onDragLeave={oDragLeave}
                        onDrop={onDrop}>

                        <div className="drop-file-input__label">
                            <img src={uploadImg} alt="upload" />
                            <p>Drag & Drop your files here</p>
                        </div>

                    <input type="file" name="image" value="" onChange={onImageDrop} />
                    
                    </div>
                )}
            {
                image ? (
                    <div className="drop-file-preview">
                        <p className="drop-file-preview__title">Ready To Predict</p>

                        <div className="drop-file-preview__item">

                            <img src={URL.createObjectURL(image)} alt="jpg" />
                            <div className="drop-file-preview__item__info">
                                <p>{(image.name)}</p>
                                <p>{(image.size / 1024).toFixed(2)} KB</p>
                            </div>

                            <span className="drop-file-preview__item__del" onClick={imageRemove}>x</span>
                        </div>

                        {/* <img src={URL.createObjectURL(image)} alt="uploaded" />
                        <span className="drop-file-preview__item__del" onClick={imageRemove}>x</span> */}
                        <div className="btn-container">
                            <button className='btnSubmit' onClick={handleClick}>submit</button>
                            <button className='btnReupload' onClick={() => setImage(null)}>re-upload</button>
                        </div>
                    </div>
                ) : null
            }
        </Wrapper>
    )
}

export default DropImage