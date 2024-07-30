import { styled } from "styled-components";

const Wrapper = styled.main`
    .drop-file-input {
    position: relative;
    width: 100%;
    height: 330px;
    border: 2px dashed var(--border-color);
    border-radius: 20px;

    display: flex;
    align-items: center;
    justify-content: center;

    background-color: var(--input-bg);
}

.drop-file-input input {
    opacity: 0;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    cursor: pointer;
}

.drop-file-input:hover,
.drop-file-input.dragover {
    opacity: 0.6;
}

.drop-file-input__label {
    text-align: center;
    color: var(--txt-second-color);
    font-weight: 600;
    padding: 10px;
}

.drop-file-input__label img {
    width: 100px;
}

.btn-container {
    display: flex;
    gap: 1rem;
}

.btn-container .btnSubmit {
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    width: 100%;
    font-family: inherit;
    font-size: 1.2rem;
    font-weight: 500;
    background-color: var(--primary-800);
    color: var(--white);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: var(--transition);
}

.btn-container .btnSubmit:hover {
    background-color: var(--primary-900);
}

.btn-container .btnReupload {
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    width: 100%;
    font-family: inherit;
    font-size: 1.2rem;
    font-weight: 500;
    background-color: var(--grey-100);
    color: var(--primary-800);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: var(--transition);
    /* box-shadow: (--shadow-3); */
    
}

.btn-container .btnReupload:hover {
    background-color: var(--grey-200);
}

.drop-file-preview {
    width: 100%;
    margin-top: 30px;
}

.drop-file-preview p {
    font-weight: 500;
}

.drop-file-preview__title {
    margin-bottom: 20px;
}

.drop-file-preview__item {
    position: relative;
    display: flex;
    margin-bottom: 10px;
    background-color: var(--input-bg);
    padding: 15px;
    border: 20px;
    border-radius: 5px;
}

.drop-file-preview__item img {
    width: 100px;
    margin-right: 20px;
    border-radius: 5px;
}

.drop-file-preview__item__info {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.drop-file-preview__item__del {
    background-color: var(--box-color);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    box-shadow: var(--box-shadow);
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.drop-file-preview__item:hover .drop-file-preview__item__del {
    opacity: 1;
}

@media (max-width: 400px) {
    .btn-container {
        flex-direction: column;
        gap: 0;
    }
    .drop-file-preview__item__info {
        font-size: 0.8rem;
        line-height: 1.5;
    }
}
`

export default Wrapper;