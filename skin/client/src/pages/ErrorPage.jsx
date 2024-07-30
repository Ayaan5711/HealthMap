import { Link } from 'react-router-dom';
import img from '../assets/images/not-found.svg';
import Wrapper from '../assets/wrappers/ErrorPage';

const ErrorPage = () => {
    return (
        <Wrapper className='container'>
            <img src={img} alt='not found' className='img' />
            <h2>Oops! Page Not Found</h2>
            <p>We can't seem to find the page you're looking for</p>
            <Link to='/'>back home</Link>
        </Wrapper>
    );
}

export default ErrorPage