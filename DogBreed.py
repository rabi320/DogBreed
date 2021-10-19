import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from PIL import Image

#title
st.markdown("![](https://i.pinimg.com/originals/a0/75/e0/a075e0909806d87578f4f5c9a8c4cdda.gif)")
st.markdown("""
# Dog Identifier 🐶

please upload a Picture of the dog you want to identify

""")

#the classes
my_classes = ['Afghan',
 'African Wild Dog',
 'Airedale',
 'American Hairless',
 'American Spaniel',
 'Basenji',
 'Basset',
 'Beagle',
 'Bearded Collie',
 'Bermaise',
 'Bichon Frise',
 'Blenheim',
 'Bloodhound',
 'Bluetick',
 'Border Collie',
 'Borzoi',
 'Boston Terrier',
 'Boxer',
 'Bull Mastiff',
 'Bull Terrier',
 'Bulldog',
 'Cairn',
 'Chihuahua',
 'Chinese Crested',
 'Chow',
 'Clumber',
 'Cockapoo',
 'Cocker',
 'Collie',
 'Corgi',
 'Coyote',
 'Dalmation',
 'Dhole',
 'Dingo',
 'Doberman',
 'Elk Hound',
 'French Bulldog',
 'German Sheperd',
 'Golden Retriever',
 'Great Dane',
 'Great Perenees',
 'Greyhound',
 'Groenendael',
 'Irish Spaniel',
 'Irish Wolfhound',
 'Japanese Spaniel',
 'Komondor',
 'Labradoodle',
 'Labrador',
 'Lhasa',
 'Malinois',
 'Maltese',
 'Mex Hairless',
 'Newfoundland',
 'Pekinese',
 'Pit Bull',
 'Pomeranian',
 'Poodle',
 'Pug',
 'Rhodesian',
 'Rottweiler',
 'Saint Bernard',
 'Schnauzer',
 'Scotch Terrier',
 'Shar_Pei',
 'Shiba Inu',
 'Shih-Tzu',
 'Siberian Husky',
 'Vizsla',
 'Yorkie']

@st.cache()
def model_loader(model_path = None, label = None):
    """
    Args:
    >model_path (str)- the path to current model.
    >label (str)-the label to predict. 
    """
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.Linear(num_features, 256),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(256, len(my_classes),                   
                          nn.LogSoftmax(dim=1)))


    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    return model.eval()

model = model_loader(model_path = "Data/model2.pt", label = my_classes)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(p = 0.5)])

uploaded_file = st.file_uploader('File uploader')

if st.button("Identify"):
    try:
        st.image(uploaded_file, width=100)

        @st.cache()
        def predictor(img, n=5):
            """
            Args: 
                >img - the image to predict.
                >n - number of top probabilities.
            
            Outputs:
                >pred - the top prediction.
                > top preds - top n predictions.
            """
            #transform the image
            img = transforms(img)
            # get the class predicted 
            pred = int(np.squeeze(model(img.unsqueeze(0)).data.max(1, keepdim=True)[1].numpy()))
            # the number is also the index for the class label
            pred = my_classes[pred]
            # get model log probabilities
            preds = torch.from_numpy(np.squeeze(model(img.unsqueeze(0)).data.numpy()))
            # convert to prediction probabilities of the top n predictions
            prob_func = nn.Softmax(dim=0)
            preds = prob_func(preds)
            top_preds = torch.topk(preds,n)
            #display at an orgenized fasion
            top_preds = dict(zip([my_classes[i]for i in top_preds.indices],[f"{round(float(i)*100,2)}%" for i in top_preds.values]))
            return pred, top_preds

        pred, preds = predictor(Image.open(uploaded_file), n=5)
        st.write("This Breed of dog is ",pred)

        df = pd.DataFrame({"Breed":list(preds.keys()),"Probability":list(preds.values())}, index = list(range(1,6)))
        st.write("Top 5 most likely dog breeds:")
        st.dataframe(df)
    except:
        st.write("no picture uploaded, please upload a picture")