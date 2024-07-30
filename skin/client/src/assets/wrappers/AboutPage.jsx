import { styled } from "styled-components";

const Wrapper = styled.section`

.about-content {
    padding: 10vh 3rem;
}
.about-title {
    font-size: 3rem;
    font-weight: 500;
    margin-bottom: 2rem;
}

.about-para {
    line-height: 1.5;
    text-align: justify;
}

@media (max-width: 400px) {
    .about-content {
        padding: 5vh 0;
    }
    .about-title {
        font-size: 2.5rem;
        text-align: center;
    }
}

`

export default Wrapper;
