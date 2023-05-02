import streamlit as st
from PIL import Image
import numpy as np
import pickle
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import joblib
import os
#print(torch.cuda.is_available())
# Load the trained model from a joblib file
#model = joblib.load("shufflemodel_dnet.pkl").cpu()
#model = joblib.load("shufflemodel_dnet.pkl")
#map_location=torch.device('cpu')
dic = {
    0: 'Alstonia Scholaris diseased',
    1: 'Alstonia Scholaris healthy',
    2: 'Arjun diseased',
    3: 'Arjun healthy',
    4: 'Bael diseased',
    5: 'Basil healthy',
    6: 'Chinar diseased',
    7: 'Chinar healthy',
    8: 'Gauva diseased',
    9: 'Gauva healthy',
    10: 'Jamun diseased',
    11: 'Jamun healthy',
    12: 'Jatropha diseased',
    13: 'Jatropha healthy',
    14: 'Lemon diseased',
    15: 'Lemon healthy',
    16: 'Mango diseased',
    17: 'Mango healthy',
    18: 'Pomegranate diseased',
    19: 'Pomegranate healthy',
    20: ' Pongamia Pinnata diseased',
    21: 'Pongamia Pinnata healthy'

}
model_path = "dnew.pth"
tmodel = models.densenet121()
tmodel.classifier = nn.Linear(in_features=1024, out_features=22, bias=True)
#print(list(tmodel.parameters())[0])
for name, child in tmodel.named_children():
   if name in ['features','avgpool','classifier']:
       #print(name + ' is unfrozen')
       for param in child.parameters():
           param.requires_grad = True
   else:
       #print(name + ' is frozen')
       for param in child.parameters():
           param.requires_grad = False


#tmodel.load_state_dict(torch.load(model_path))
tmodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#print(list(tmodel.parameters())[0])
# Set page title and favicon
st.set_page_config(page_title="Image Input Example", page_icon=":camera_flash:")






# Set page style
st.markdown(
    """
    <style>
    .stApp {
        background-color:  ##5f0ba3;
    }
    .stTextInput>label {
        color: #333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set page header
st.title("Welcome to the leaf disease detection system")
st.write("Upload an image of a leaf below:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
def load_model(model_name):
    if model_name == 'Model 1':
        model = tmodel
    elif model_name == 'Model 2':
        model = joblib.load('model2.pkl')
    else:
        model = joblib.load('model3.pkl')
    return model

# Create a drop-down menu to select the model
#model_name = st.selectbox('Select a model', ['Model 1', 'Model 2', 'Model 3'])
def open_html_file():
    file_path ='ab.html'
    os.system(f'start {file_path}') # Windows
    # os.system(f'open {file_path}') # Mac

# Add button to open HTML file
if st.button('Open HTML File'):
    open_html_file()
if uploaded_file is not None:
    # Load image

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    #image = image.convert('RGB')
    image = image.resize((64, 64))

    # Preprocess image
    img_array = np.array(image,dtype='float32')
    print(img_array.shape)
    img_array = img_array / 255.0
    #img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array.reshape(1,3,64,64)
    # Make prediction
    #img_array = torch.from_numpy(img_array)
    img_array = img_array.reshape(3,64,64)
    img_array = torch.tensor(img_array)
    img_array = img_array.unsqueeze(0)

    imgs = torch.zeros((128, 3, 64, 64))
    lbls = torch.ones((1))
    imgs[0] = img_array

    imgs = TensorDataset(img_array,lbls)
    er = DataLoader(imgs,shuffle=0,batch_size=1)
    for i,j in er:
        tmodel.eval()
        prediction = tmodel(i)
        probs = F.softmax(prediction, dim=1)
        max_prob, preds = torch.max(probs, dim=1)
        break

    #prediction = tmodel(imgs)

    #print(preds.item())
    # Print prediction
    st.write(f"Prediction: {dic[preds[0].item()]}")
    print(preds[0]+1)

    # Display image
    #st.image(image, caption="Uploaded Image", use_column_width=True)
