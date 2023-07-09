## Imported Modules ##
import streamlit as st
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import os
import copy

## Functions ##

def image_loader(image_name):
    image = Image.open(image_name)
    # Fake batch dimension required to fit netowrk's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device,torch.float)

def show_img(tensor,title=None,num=0):
    # Reconverts into PIL Image
    unloader = transforms.ToPILImage()
    # Clone the tensor to not do changes on it 
    image = tensor.cpu().clone()
    # Remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)
    
    if num==1:
        return image

def gram_matrix(input):
    # Here a= batch size(=1), b = Number of Feature Maps
    # (c,d)= Dimensions of a Feature Map
    a,b,c,d =input.size()
    
    # Resizing the features of an Lth layer
    features =input.view(a*b,c*d)
    # Computing the Gram Product
    G =torch.mm(features, features.t())
        
    return G.div(a*b*c*d)

@st.cache(allow_output_mutation = True) 
def load_VGG19():
    cnn= models.vgg19(pretrained=True).features.eval()
    return cnn

# Function to get model and losses
def get_style_model_losses(cnn,norm_mean,norm_std,style_img,content_img):
    content_layers=content_default
    style_layers= style_default
    # Using the normalization module
    cnn = copy.deepcopy(cnn)
    normalization =Normalization(norm_mean,norm_std).to(device)
    
    # Losses
    content_losses=[]
    style_losses=[]
    
    model = nn.Sequential(normalization)
    i=0
    
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name='relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name='pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name='bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized Layer:{}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)
        
        # Add the Content Loss Layers
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i),content_loss)
            content_losses.append(content_loss)
        
        # Add the Style Loss Layers
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)
        
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
            break
    model =model[:(i+1)]
    
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn,norm_mean,norm_std,content_img,style_img,input_img,num_steps=600,style_weight=1000000,content_weight=1):
    print("Building the Style Transfer Model..")
    
    model,style_losses,content_losses=get_style_model_losses(cnn,norm_mean,norm_std,style_img,content_img)
    
    # We want to optimize the input and not the model parameters 
    input_img.requires_grad_(True)
    # Putting the model in evaluation mode
    model.eval()
    model.requires_grad_(False)
    
    optim = get_input_optimizer(input_img)
    
    print("Optimizing")
    
    
    run =[0]
    
    while run[0]<=num_steps:
            
        def closure():
            # Correct the values of the updated Input Image
            with torch.no_grad():
                input_img.clamp_(0,1)
            optim.zero_grad()
            model(input_img)
            style_score=0
            content_score=0
                
            # As per the formula to calculate loss using Style and Content Losses
            for sl in style_losses:
                style_score+=sl.loss
            for cl in content_losses:
                content_score+=cl.loss
                
            style_score*=style_weight
            content_score*=content_weight
                
            loss = style_score+content_score
            loss.backward()
                
            run[0]+=1
            if run[0]%50==0:
                print("run {}".format(run))
                print('Style Loss: {:4f} Content Loss: {:4f}\n'.format(style_score.item(),content_score.item()))
                print()
            return style_score+content_score

        optim.step(closure)
        
    with torch.no_grad():
        input_img.clamp_(0,1)
    return input_img

def run(content_img,style_img,input_img,num=600):
    
    cnn_mean = torch.tensor([0.485,0.456,0.406])
    cnn_std = torch.tensor([0.229,0.224,0.225])

    output =run_style_transfer(cnn,cnn_mean,cnn_std,content_img,style_img,input_img,num_steps=num)

    
    img=show_img(output,num=1)
    return img

# Custom Class and NN Modules

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Detach the target content from the tree used
        # To dynamically compute the gradient: This is a stated value
        self.target =target.detach()
        
    def forward(self,input):
        self.loss =F.mse_loss(input,self.target)
        return input

# Style Loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Detach the target content from the tree used
        # To dynamically compute the gradient: This is a stated value
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self,input):
        G= gram_matrix(input)
        self.loss =F.mse_loss(G,self.target)
        return input





# Module to normalize input images
class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        # Reshape the mean and std so that they can directly work with image Tensor.
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
        
    def forward(self,img):
        # Normalize img
        return (img-self.mean)/self.std

# Variables

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = (512,512) if torch.cuda.is_available() else (128,128)
loader = transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])
cnn_mean = torch.tensor([0.485,0.456,0.406])
cnn_std = torch.tensor([0.229,0.224,0.225])
# Desired depth layers to compute content and style losses
content_default = ['conv_4']
style_default =['conv_1','conv_2','conv_3','conv_4','conv_5']

    


       
## Layout##
rad = st.sidebar.radio("Navigation",["Home","Image Generation"])


if rad=="Home":
    st.title("Neural Style Transfer")
    st.subheader("By Hardik Pahwa")
    st.markdown("""""",True)
    
    st.image("Styled_hardik.jpg")
    
elif rad=="Image Generation":
    st.write("""
            # Image Generation
            The results generated may not always be perfectly generated            
    """)
    
    
    v1 = st.radio("Choose Style of Image",["Candy","Picasso","Spiderman","Starry_Night","Mosaic"],index=3)
    
    if v1=="Candy":
        style_img =image_loader("data/style-images/"+"candy.jpg")
        st.image("data/style-images/"+"candy.jpg",width=400)
    elif v1=="Picasso":
        style_img =image_loader("data/style-images/"+"picasso.jpg")
        st.image("data/style-images/"+"picasso.jpg",width=400)
    elif v1=="Spiderman":
        style_img =image_loader("data/style-images/"+"spiderman.jpg")
        st.image("data/style-images/"+"spiderman.jpg",width=400)
    elif v1=="Starry_Night":
        style_img =image_loader("data/style-images/"+"vg_starry_night.jpg")
        st.image("data/style-images/"+"vg_starry_night.jpg",width=400)
    elif v1=="Mosaic":
        style_img =image_loader("data/style-images/"+"mosaic.jpg")
        st.image("data/style-images/"+"mosaic.jpg",width=400)
    
    file = st.file_uploader("Upload an Image")
    st.write(file)
    st.set_option('deprecation.showfileUploaderEncoding', False)
        
    if file != None:
        with st.spinner("Generating Image..."):
            image = Image.open(file)
            
            content_img = image_loader(file)
            assert style_img.size()==content_img.size(),"Style and Content Images must be of Same Size"
            
            cnn = load_VGG19()
            input_img = content_img.clone()
            
            st.image(file,width=512)
            
            result = run(content_img,style_img,input_img)
            st.image(result,width=512)
    else:
        st.write("No file uploaded")
