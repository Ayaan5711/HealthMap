import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

class ImagePrediction:

    def chest(self, image_path):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 7)
        model.load_state_dict(torch.load('src/image_models/chest_resnet50_model_state_dict.pth', map_location=torch.device('cpu')))
        model.eval()

        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('L')
        image = Image.merge("RGB", (image, image, image))
        image = preprocess(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        with open("src/classnames/chest.txt") as file:
            data = file.readlines()
        classes = []
        for i in data:
            classes.append(i.strip("\n"))

        return classes[predicted.item()]
    



    def brain(self, image_path):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load('src/image_models/brain_resnet50_model_state_dict.pth', map_location=torch.device('cpu')))
        model.eval()

        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('L')
        image = Image.merge("RGB", (image, image, image))
        image = preprocess(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        with open("src/classnames/brain.txt") as file:
            data = file.readlines()
        classes = []
        for i in data:
            classes.append(i.strip("\n"))

        return classes[predicted.item()]
    




    def malaria(self, image_path):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
            nn.Softmax(dim=1)
        )
        model.load_state_dict(torch.load('src/image_models/malaria_resnet152_model_state_dict.pth', map_location=torch.device('cpu')))
        model.eval()

        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('L')
        image = Image.merge("RGB", (image, image, image))
        image = preprocess(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)

        with open("src/classnames/malaria.txt") as file:
            data = file.readlines()
        classes = []
        for i in data:
            classes.append(i.strip("\n"))

        return classes[predicted.item()]
    



    def skin(self, image_path):
        model = models.googlenet(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
            nn.Softmax(dim=1)
        )
        model.load_state_dict(torch.load('src/image_models/skin_googlenet_model_state_dict.pth', map_location=torch.device('cpu')))
        model.eval()

        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('L')
        image = Image.merge("RGB", (image, image, image))
        image = preprocess(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)

        with open("src/classnames/skin.txt") as file:
            data = file.readlines()
        classes = []
        for i in data:
            classes.append(i.strip("\n"))

        return classes[predicted.item()]





    def predict(self,image_path):

        # Load the trained model
        model = models.inception_v3(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)


        model.load_state_dict(torch.load('src/image_models/brain_chest_malaria_skin_inception_v3_model_state_dict.pth', map_location=torch.device('cpu')))
        model.eval()

        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('L')
        image = Image.merge("RGB", (image, image, image))
        image = preprocess(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            prob = probabilities[0][predicted].item()
            if prob > 0.3:
                if predicted.item() == 0:
                    prediction = "Brain"
                    chest_pred = self.brain(image_path=image_path)
                    return chest_pred,prediction

                elif predicted.item() == 1:
                    prediction = "Chest"
                    brain_pred = self.chest(image_path=image_path)
                    return brain_pred,prediction

                elif predicted.item() == 2:
                    prediction = "Malaria"
                    malaria_pred = self.malaria(image_path=image_path)
                    return malaria_pred,prediction
                
                elif predicted.item() == 3:
                    prediction = "Skin"
                    skin_pred = self.skin(image_path=image_path)
                    return skin_pred,prediction

            else:
                return "Sorry!, we don't have that disease in out database."

        