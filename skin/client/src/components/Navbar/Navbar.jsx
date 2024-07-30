import { useRef } from 'react'
// import { navLinks } from '../../data'
import { FaBars } from "react-icons/fa";
import { useNavigate, Link } from 'react-router-dom'
import './Navbar.css'
import { useState } from 'react'

const Navbar = () => {

    const [showLinks, setShowLinks] = useState(false)
    const linksContainerRef = useRef(null)
    const linksRef = useRef(null)

    const linkStyles = {
        height: showLinks ? `${linksRef.current.getBoundingClientRect().height}px` : '0px',
    }

    const navigate = useNavigate()

    const handleClick = () => {
        navigate("/upload_img")
    }

    return (
        <nav>
            <div className="nav-center">
                <div className="nav-header">
                    <h2 className="logo"><Link to="/">Dermnet</Link></h2>
                    <button className="nav-toggle"
                        onClick={() => setShowLinks(!showLinks)}
                    >
                        <FaBars />
                    </button>
                </div>
                <div className="links-container" ref={linksContainerRef} style={linkStyles}>
                    <ul className='links' ref={linksRef}>
                        <li><Link to="/about_us">about us</Link></li>
                        <li><button type='button' className='btnPredict' onClick={handleClick}>Predict</button></li>
                    </ul>
                </div>
            </div>
        </nav>
    )
}

export default Navbar