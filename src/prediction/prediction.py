from keras.models import load_model
from PIL import Image, ImageOps 
import numpy as np

class ImageClassification:
    def __init__(self, image_path):
        self.result = ""
        self.image_path = image_path

    def chest_predict(self):

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(self.image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        chest_model = load_model("src/models/Chest_Model.h5", compile=False)
        chest_class_names = ["Atelectasis","Effusion","Infiltration","Nodule","Pleural Thickening","Pneumonia","Pneumothorax"]

        chest_prediction = chest_model.predict(data)
        chest_index = np.argmax(chest_prediction)
        chest_class_name = chest_class_names[chest_index]
        confidence_score = chest_prediction[0][chest_index]

        if confidence_score > 0.5:
            return f"This X-Ray is predicted to have {chest_class_name}, Please Consult Doctor."
        else:
            return "This X-Ray does not Brain Disease, or we don't have it in our database."
    

    def brain_predict(self):

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(self.image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
 
        brain_model = load_model("src/models/Brain_Model.h5", compile=False)
        brain_class_names = ["Glioma","Meningioma","Pituitary"]

        brain_prediction = brain_model.predict(data)
        brain_index = np.argmax(brain_prediction)
        brain_class_name = brain_class_names[brain_index]
        confidence_score = brain_prediction[0][brain_index]

        if confidence_score > 0.5:
            return f"This X-Ray is predicted to have {brain_class_name}, Please Consult Doctor."
        else:
            return "This X-Ray does not Brain Disease, or we don't have it in our database."
    


    def malaria_predict(self):

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(self.image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        model = load_model("src/models/malaria.h5")
        class_names = ['yes','no']

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        if confidence_score > 0.5:
            return f"This X-Ray is predicted to have {class_name}, Please Consult Doctor."
        else:
            return "This X-Ray does not show any sign of Malaria."