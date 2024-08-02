# HealthMap

HealthMap is an advanced platform designed for personalized medicine and bioinformatics analysis. It predicts diseases based on symptoms, identifies specific diseases, and offers personalized medicine for patient drug responses. The platform leverages state-of-the-art large language models (LLMs) and vision-language models (VLMs) to enhance the knowledge domain and streamline healthcare processes.

## Features

- **Disease Prediction**: Predicts potential diseases based on user-provided symptoms.
- **Disease Identification**: Identifies a particular disease from a list of symptoms, offering detailed information and recommended actions.
- **Personalized Medicine**: Provides personalized drug responses for patients, considering factors like age, gender, and medical history.
- **Knowledge Enhancement**: Utilizes cutting-edge LLMs and VLMs to provide comprehensive medical knowledge and assist healthcare professionals.
- **User-Friendly Interface**: Easy-to-navigate interface for both patients and healthcare providers.

- Demo Video **[Link](https://drive.google.com/file/d/1ZWa58DOyrsY8Mk9yHauWxg_6hak5APEL/view)**

## Getting Started

### Installation and Setup:

1. Make Sure You are using `python==3.11`

2. Clone the repository:
    ```sh
    git clone https://github.com/Ayaan5711/healthmap.git
    ```
3. Navigate to the project directory:
    ```sh
    cd healthmap
    ```

4. Create a New Virtual Environment:
    - 1. Creating a new Environment:
        ```sh
        python -m venv HealthMapvenv
        ```
    - 2. Activating HealthMapvenv Environment:
        ```sh
        HealthMapvenv\Scripts\activate
        ```

5. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

6. Download [this folder](https://drive.google.com/drive/folders/17bhvoDmlmTXdryUS-8AGeDieWOtT4q31?usp=sharing) from drive and store it inside src folder in this directory"

7. Make sure your directory should look like this
    ```sh
    src/
        - image_models/
            - brain_chest_malaria_skin_inception_v3_model_state_dict.pth 
            - ....etc
        - ...other files...
    ... other files ...
    ```


8. Start the application:
    ```sh
    python app.py
    ```
9. Open your web browser and navigate to `http://localhost:5000`.

10. Follow the on-screen instructions to use HealthMap's features.

## Components

- **Disease Prediction Module**: Analyzes user symptoms and predicts possible diseases.
- **Disease Identification Module**: Identifies specific diseases based on detailed symptom input.
- **Personalized Drug Response Module**: Predicts patient-specific drug responses using multi-omic data.
- **LLMs and VLMs Integration**: Enhances the platform’s capabilities with advanced language and vision models.

## Contributing

We welcome contributions to HealthMap! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact us at:

- Email: ahmedayaan570@gmail.com
- LinkedIn: [Ayaan Ahmed](https://www.linkedin.com/in/ayaan-ahmed-70a5b0157/)
- GitHub: [Ayaan5711](https://github.com/Ayaan5711)

