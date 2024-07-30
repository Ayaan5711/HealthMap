import React from 'react'
import Navbar from '../components/Navbar/Navbar'
import Wrapper from '../assets/wrappers/HomePage'
import banner from '../assets/images/banner_img.jpg'
import { useNavigate } from 'react-router-dom'

const Home = () => {

    const navigate = useNavigate()
    
    return (
        <Wrapper className='container'>
            <Navbar />
            <section className="hero">
                <div className="hero-content">
                    <h1 className="title">your health is our top priority</h1>
                    <p className='para'>Lorem ipsum dolor sit amet consectetur adipisicing elit. Vitae quasi earum officia placeat ad magnam nisi vero, dicta saepe iste!</p>
                    <div className="btn-container">
                        <button className="btnPredict" type='button' onClick={()=>navigate("/upload_img")}>you wanna predict?</button>
                    </div>
                </div>
                <div className="hero-img">
                    <img src={banner} title='skin-disease' alt="banner-img" className='img' />
                </div>
            </section>
        </Wrapper>
    )
}

export default Home