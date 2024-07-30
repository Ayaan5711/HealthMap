import React, { useEffect, useState } from 'react'
import Navbar from '../components/Navbar/Navbar';
import DropImage from './DropImage';
import Wrapper from '../assets/wrappers/UploadImage';

const UploadPage = () => {

    const [isLoading, setIsLoading] = useState(true)
    const [showBox, setShowBox] = useState(true)
    const [context, setContext] = useState({})
    const [image, setImage] = useState(null);

    useEffect(() => {
        setTimeout(() => {
            setIsLoading(false)
        }, 1500)
    }, [showBox])

    if (isLoading) {
        return (
            <Wrapper className="container">
                <Navbar />
                <div className="loading"></div>
            </Wrapper>
        )
    }

    const onFileChange = (image) => {
        console.log(image);
        setImage(image)
    }

    return (
        <Wrapper className="container">
            <Navbar />
            {showBox ? (
                <div className="box">
                    <h2 className="header">
                        upload the image
                    </h2>
                    <p className='header-info'>Upload the image to predict what's the disease</p>
                    <DropImage onFileChange={onFileChange} setShowBox={setShowBox} setIsLoading={setIsLoading} setContext={setContext} />
                </div>

            ) : (
                <>
                    <div className="title-container">
                        <div className="img-container">
                            <img src={URL.createObjectURL(image)} alt="image" width='100px' />
                            <div className="disease-title">
                                <h3>predicted disease </h3>
                                <p>{context.predictedLabel}</p>
                            </div>
                        </div>
                        <button className='btnReupload' onClick={() => setShowBox(true)}>re-upload</button>
                    </div>
                    <div className="prediction-container">
                        <div className="side-container">
                            <div className="disease-summary m-bottom infoBox">
                                <h3>summary about the disease : </h3>
                                <p>{context.about}</p>
                            </div>
                            <div className="disease-medication m-bottom infoBox">
                                <h3>recommended medication : </h3>
                                <p>{context.medicine}</p>
                            </div>
                        </div>
                        <div className="side-container">
                            <div className="twitter-api m-bottom infoBox">
                                <h3>home remedies : </h3>
                                <p>{context.remedies}</p>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </Wrapper>
    )
}

export default UploadPage