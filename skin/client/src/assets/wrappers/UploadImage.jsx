import { styled } from "styled-components";

const Wrapper = styled.main`

    .btnReupload {
        padding: 0.7rem 1rem;
        font-family: inherit;
        font-weight: 500;
        background-color: #127c12;
        color: var(--grey-100);
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: var(--transition);        
    }

    .btnReupload:hover {
        background-color: #0d4f0d;
    }

    .title-container {
        background-color: var(--box-bg);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: var(--shadow-2);
        margin-top: 3vh;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .img-container {
        display: flex;
        align-items: center;
    }

    .title-container img {
        width: 100px;
        margin-right: 1.2rem;
        border-radius: 5px;
    }

    .title-container p {
        margin-left: 0.2rem;
        text-transform: capitalize;
    }

    .disease-title {
        h3,p{
            font-size: 1.2rem;
            display: inline;
        }
    }

    .prediction-container {
        width: 100%;
        /* min-height: 60vh; */
        margin-top: 3vh;
        display: flex;
        gap: 2rem;
    }

    .side-container {
        width: 50%;
    }

    .infoBox {
        background-color: var(--box-bg);
        padding: 1.3rem;
        border-radius: 10px;
        box-shadow: var(--shadow-2);
        h3 {
            font-size: 1.2rem;
        }
    }
    
    .disease-summary, .disease-medication, .twitter-api {
        h3 {
            margin-bottom: 0.8rem;
        }
        p{
            font-size: 1rem;
            text-align: justify;
            line-height: 2;
        }
    }

    @media (max-width: 768px) {

        .prediction-container {
            flex-direction: column;
            gap: 0;
        }
        .side-container {
            width: 100%;
        }

        .title-container {
            align-items: start;
            h3, p {
                font-size: 1.1rem;
                display: block;
                line-height: 1.5;
            }
            p {
                margin-left: 0;
            }
        }
        .btnReupload {
            align-self: center;
        }
    }

    @media (max-width: 400px) {
        .btnReupload {
            width: 100%;
        }

        .title-container, .img-container {
            flex-direction: column;
            gap: 1rem;
        }

        .title-container {
            h3,p {
                font-size: 1rem;
                display: block;
                line-height: 1.5;
            }
            p {
                margin-left: 0;
            }
        }
        
        .img-container {
            align-items: center;
        }

        .img-container img{
            align-self: center;
            margin-right: 0;
            width: 50%;
        }
        .disease-title {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .disease-summary, .disease-medication, .twitter-api {
            h3 {
                font-size: 1rem;
            }
            p{
                line-height: 1.7;
                text-align: justify;
            }
        }
    }
`

export default Wrapper;