import tensorflow as tf
import numpy as np

class ImagePrediction:

    def predict(self,image_path):

        result = ""
        img=tf.keras.preprocessing.image.load_img(image_path,target_size = (224, 224))

        model = tf.keras.models.load_model('image_models/Chest_Brain_ResNet50_V2.h5')

        x=tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add a batch dimension
        x = x / 255.0
        prediction = model.predict(x)
        index = np.argmax(prediction)
        class_names = ["Chest","Brain"]
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        if class_name == 'Chest' and confidence_score >= 0.85:
            chest_model = tf.keras.models.load_model('image_models/ChestV3_ResNet201_V2.h5')
            chest_prediction = chest_model.predict(x)
            chest_index = np.argmax(chest_prediction)
            chest_class_names = ["Atelectasis","Effusion","Infiltration","Nodule","Pleural Thickening","Pneumonia","Pneumothorax"]
            chest_class_name = chest_class_names[chest_index]
            chest_confidence_score = prediction[0][chest_index]
            if chest_confidence_score >= 0.3:
                result = chest_class_name
            else:
                result = "None"

            return result, chest_class_name
            



        
        elif class_name == 'Brain' and confidence_score >= 0.85:
            brain_model = tf.keras.models.load_model('image_models/BrainV3_ResNet201.h5')
            brain_prediction = brain_model.predict(x)
            brain_index = np.argmax(brain_prediction)
            brain_class_names = ["Glioma","Meningioma","Pituitary"]
            brain_class_name = brain_class_names[index]
            brain_confidence_score = brain_prediction[0][brain_index]
            if brain_confidence_score >= 0.4:
                result = brain_class_name
            else:
                result = "None"

            return result, brain_class_name
            




        else:
            result = "None"
            return result , "None"