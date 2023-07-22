from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import *
import os
import sys
import argparse
from PIL import Image
import cv2
import time
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import torch
import torchvision.models as models
import cityscapesscripts
from torchvision.models import ResNet
import json
import subprocess

#st.set_page_config(layout = "wide")
st.set_page_config(page_title = "PEDESTRIAN DETECTION - PSISAV", page_icon="👨🏼‍🦯")

#Load the Citypersons Yolov5 trained model
#model1=torch.load('/Users/cs/Desktop/PSISAV/Dataset/psisav.pth')

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown(
    
    '''<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

#################### Title #####################################################

st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>PSISAV Pedestrian detection</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Developed by team PSISAV</h2>", unsafe_allow_html=True)
#st.markdown('---') # inserts underline
#st.markdown("<hr/>", unsafe_allow_html=True) # inserts underline
st.markdown('#') # inserts empty space

#--------------------------------------------------------------------------------

DEMO_VIDEO = os.path.join('data', 'videos', 'sampleVideo0.mp4')
DEMO_PIC = os.path.join('data', 'images', 'bus.jpg')

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

#---------------------------Main Function for Execution--------------------------

def main():

    source = ("Detect From Image", "Detect From Video")
    source_index = st.sidebar.selectbox("Select Activity", range(
        len(source)), format_func = lambda x: source[x])
    
    cocoClassesLst = ["person"]
    #cocoClassesLst = ["person","traffic light","stop sign"]

    #classes_index = [0,9,11]
    #classes_index=[0,1,2]
    classes_index=[0]

    
    isAllinList = 80 in classes_index
    if isAllinList == True:
        classes_index = classes_index.clear()
        
    print("Selected Classes: ", classes_index)
    
    #################### Parameters to setup ########################################
    # MAX_BOXES_TO_DRAW = st.sidebar.number_input('Maximum Boxes To Draw', value = 5, min_value = 1, max_value = 5)
    #deviceLst = ['cpu', '0', '1', '2', '3']
    deviceLst = ['cpu']
    DEVICES = st.sidebar.selectbox(" ", deviceLst, index = 0)
    print("Devices: ", DEVICES)
    #MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4)

    MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.2)

    #################### Parameters to setup Streamlit and model.pth ########################################
    
    weights = os.path.join("weights", "yolov5s.pt")

    if source_index == 0:
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type = ['png', 'jpeg', 'jpg'])
        
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Resource Loading...'):
                st.sidebar.text("Uploaded Pic")
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture.save(os.path.join('data', 'images', uploaded_file.name))
                data_source = os.path.join('data', 'images', uploaded_file.name)
        
        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("DEMO Pic")
            st.sidebar.image(DEMO_PIC)
            data_source = DEMO_PIC
        
        else:
            is_valid = False
    
    elif source_index == 1:
        
        uploaded_file = st.sidebar.file_uploader("Upload Video", type = ['mp4'])
        
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Resource Loading...'):
                st.sidebar.text("Uploaded Video")
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                data_source = os.path.join("data", "videos", uploaded_file.name)
        
        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("DEMO Video")
            st.sidebar.video(DEMO_VIDEO)
            data_source = DEMO_VIDEO
        
        else:
            is_valid = False
    
    else:
        ######### Select and capture Camera (Disabled) ################# Disabled In the code

        

        
        selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "Use Other Camera"), index = 0)
        if selectedCam:
            if selectedCam == "Use Other Camera":
                data_source = int(1)
                is_valid = True
            else:
                data_source = int(0)
                is_valid = True
        else:
            is_valid = False
        
        st.sidebar.markdown("<strong>Press 'q' multiple times on camera window and 'Ctrl + C' on CMD to clear camera window/exit</strong>", unsafe_allow_html=True)
        
    if is_valid:
        print('valid')
        if st.button('Detect'):
            if classes_index:
                with st.spinner(text = 'Inferencing, Please Wait.....'):
                    run(weights = weights, 
                        source = data_source,  
                        #source = 0,  #for webcam
                        conf_thres = MIN_SCORE_THRES,
                        #max_det = MAX_BOXES_TO_DRAW,
                        device = DEVICES,
                        save_txt = True,
                        save_conf = True,
                        classes = classes_index,
                        nosave = False, 
                        )
                        
            else:
                with st.spinner(text = 'Inferencing, Please Wait.....'):
                    run(weights = weights, 
                        source = data_source,  
                        #source = 0,  #for webcam
                        conf_thres = MIN_SCORE_THRES,
                        #max_det = MAX_BOXES_TO_DRAW,
                        device = DEVICES,
                        save_txt = True,
                        save_conf = True,
                    nosave = False, 
                    )
                    
            
                

            if source_index == 0:
                with st.spinner(text = 'Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                            pathImg = os.path.join(get_detection_folder(), img)
                            st.image(pathImg)
                            #image_with_boxes = draw_bounding_boxes(img, boxes, labels)
                    
                    st.markdown("### Output")
                    st.write("Path of Saved Images: ", pathImg)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))  
                    
                    
            elif source_index == 1:
                with st.spinner(text = 'Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            #st.video(os.path.join(get_detection_folder(), vid))
                            #video_file = open(os.path.join(get_detection_folder(), vid), 'rb')
                            #video_bytes = video_file.read()
                            #st.video(video_bytes)
                            video_file = os.path.join(get_detection_folder(), vid)
                            
                stframe = st.empty()
                cap = cv2.VideoCapture(video_file)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Width: ", width, "\n")
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("Height: ", height, "\n")

                while cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                        
                        #stframe.image(cv2.resize(img, (width, height)), channels='BGR', use_column_width=True)
                    else:
                        break
                
                cap.release()
                st.markdown("### Output")
                st.write("Path of Saved Video: ", video_file)    
                st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))    
                
            
            else:
                with st.spinner(text = 'Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            liveFeedvideoFile = os.path.join(get_detection_folder(), vid)
                    
                    st.markdown("### Output")
                    st.write("Path of Live Feed Saved Video: ", liveFeedvideoFile)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))    
                  
                


# --------------------MAIN FUNCTION CODE------------------------                                                                    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------


