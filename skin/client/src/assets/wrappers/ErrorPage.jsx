import { styled } from "styled-components";

const Wrapper = styled.main`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    .img {
        width: 30vw;
        margin-top: 10vh;
        margin-bottom: 5rem;
    }
    h2 {
        line-height: 2;
    }
    p {
        line-height: 1.5;
        margin-bottom: 2rem;
    }
    a {
        color: var(--primary-500);
    }
`

export default Wrapper