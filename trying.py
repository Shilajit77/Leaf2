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
#--------------------model1------------------------------#
model_path = "models/dnew.pth"
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

tmodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

tmodel2 = models.densenet121()
tmodel2.classifier = nn.Linear(in_features=1024, out_features=22, bias=True)
tmodel2.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model_path = "models/mnet.pth"
model2 = models.mobilenet_v2(pretrained=False)
model2.classifier = nn.Linear(in_features=1280, out_features=22, bias=True)
model2.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))








from collections import Counter
def most_frequent(List):
    return max(set(List), key = List.count)



st.set_page_config(page_title="Image Input Example", page_icon=":camera_flash:")

# Functions:

def preprocess_img(image):
    image = image.resize((64, 64))

    # Preprocess image
    img_array = np.array(image, dtype='float32')
    print(img_array.shape)
    img_array = img_array / 255.0
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array.reshape(1,3,64,64)
    # Make prediction
    # img_array = torch.from_numpy(img_array)
    img_array = img_array.reshape(3, 64, 64)
    img_array = torch.tensor(img_array)
    img_array = img_array.unsqueeze(0)

    imgs = torch.zeros((128, 3, 64, 64))
    lbls = torch.ones((1))
    imgs[0] = img_array

    imgs = TensorDataset(img_array, lbls)
    er = DataLoader(imgs, shuffle=0, batch_size=1)
    return er

def prediction(er,tmodel):
    for i,j in er:
        tmodel.eval()
        prediction = tmodel(i)
        probs = F.softmax(prediction, dim=1)
        max_prob, preds = torch.max(probs, dim=1)
        break
    return preds

def load_model(model_name):
    if model_name == 'Model 1 (DenseNet)':
        model = tmodel
    elif model_name == 'Model 2 (Resnet)':
        model = tmodel2
    elif model_name == 'Model 3 (VGG16)':
        model = model2
    elif model_name == 'Ensemble (Best One)':
        model = ensemble(er,tmodel,tmodel2,model2)
    return model

def ensemble(er,model1,model2,model3):
    predictions1 = []
    predictions2 = []
    predictions3 = []
    max1 = []
    max2 = []
    max3 = []
    for i,j in er:
        model1.eval()
        prediction = model1(i)
        probs = F.softmax(prediction, dim=1)
        max_prob, preds1 = torch.max(probs, dim=1)
        predictions1.append(preds1)
        max1.append(max_prob)
        break
    for i,j in er:
        model2.eval()
        prediction = model2(i)
        probs = F.softmax(prediction, dim=1)
        max_prob, preds2 = torch.max(probs, dim=1)
        predictions2.append(preds2)
        max2.append(max_prob)
        break
    for i,j in er:
        model3.eval()
        prediction = model3(i)
        probs = F.softmax(prediction, dim=1)
        max_prob, preds2 = torch.max(probs, dim=1)
        predictions3.append(preds2)
        max3.append(max_prob)
        break
    p1 = []
    for i in predictions1:
        for j in i:
            p1.append(j.item())
    p2 = []
    for i in predictions2:
        for j in i:
            p2.append(j.item())
    p3 = []
    for i in predictions3:
        for j in i:
            p3.append(j.item())
    m1 = []
    m2 = []
    m3 = []
    for i in max1:
        for j in i:
            m1.append(j.item())
    for i in max2:
        for j in i:
            m2.append(j.item())
    for i in max3:
        for j in i:
            m3.append(j.item())

    final_pred = []
    for i in range(len(p1)):
        compare = []
        prob = []
        compare.append(p1[i])
        compare.append(p2[i])
        compare.append(p3[i])
        prob.append(m1[i])
        prob.append(m2[i])
        prob.append(m3[i])
        a = most_frequent(compare)
        d = Counter(compare)
        count = d[a]
        if (count == 1):
            maximum = np.argmax(prob, 0)
            final_pred.append(compare[maximum])
        else:
            final_pred.append(a)
    return final_pred
















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
st.title("Welcome to the Leaf Disease Detection System")
st.write("Upload an image of a leaf below:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# Create a drop-down menu to select the model
model_name = st.selectbox('Select a model', ['Model 1 (DenseNet)', 'Model 2 (Resnet)', 'Model 3 (VGG16)','Ensemble (Best One)'])
def open_html_file():
    file_path ='credit.html'
    os.system(f'start {file_path}') # Windows
    # os.system(f'open {file_path}') # Mac

# Add button to open HTML file
if st.button('Go to Credit Page'):
    open_html_file()
if uploaded_file is not None:
    # Load image

    image = Image.open(uploaded_file)
    er = preprocess_img(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if(model_name=='Ensemble (Best One)'):
        preds = load_model(model_name)
        st.write(f"Prediction: {dic[preds[0]]}")
    else:
        model = load_model(model_name)

        preds = prediction(er,model)

        st.write(f"Prediction: {dic[preds[0].item()]}")
        print(preds[0]+1)
        st.write(f'Selected model is: {model_name}')
