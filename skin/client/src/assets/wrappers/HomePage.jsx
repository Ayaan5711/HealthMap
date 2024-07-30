import { styled } from "styled-components";

const Wrapper = styled.main`
    .hero {
        display: flex;
        align-items: center;
        margin-top: 1rem;
        min-height: 78vh;
        /* border: 1px solid; */
    }
    .hero-content {
        padding: 1rem 2rem;
        width: 50%;
    }
    .title {
        font-weight: 400;
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .para { 
        line-height: 1.8;
        margin-bottom: 2rem;
    }
    .hero-img {
        width: 50%;
        padding: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-img .img {
        width: 100%;
        border-radius: 40px;
        box-shadow: var(--shadow-2);
        object-fit: cover;
    }

    .btnPredict:hover {
        box-shadow: var(--shadow-4);
    }

    @media (max-width: 768px) {
        .hero {
            flex-direction: column;
        }
        .hero-content {
            width: 100%;
            margin-block: 2rem;
        }
        .title {
            font-size: 3.5rem;
        }
        .hero-img {
            width: 100%;
        }

    }
    @media (max-width: 400px) {
        .hero-content {
            padding-inline: 0;
        }
        .hero-img {
            padding: 0;
        }
        .img {
            border-radius: 1rem;
        }
        .title {
            font-size: 2rem;
            /* text-align: center; */
        }
    }
`

export default Wrapper;