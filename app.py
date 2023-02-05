# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    app.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/01/05 18:15:40 by ebennace          #+#    #+#              #
#    Updated: 2023/01/10 17:24:31 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from PIL import Image, ImageDraw
import turtle
import random
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time
from deep_translator import GoogleTranslator
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image
from PIL.Image import fromarray

model = load_model("best_model")


def load_class_names():
    my_file = open("class_names.txt", "r")
    data = my_file.read()
    class_names = data.split("\n")
    my_file.close()
    return (class_names)

def preprocessing_image(draw, img_size=28):
    
    img_pil = fromarray(draw)
    img_resize = img_pil.resize((28, 28), Image.ANTIALIAS)
    img = np.array(img_resize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    result = resized[:, :, np.newaxis]
    st.image(result)
    st.write(result.shape)
    # img = img/255.
    
    return (result)

st.set_page_config("Deep Pictionary", "✏")

st.header("Deep Pictionary 🧠")
st.image("pictionary.jpeg")


# with st.sidebar:
    
#     result = st.selectbox("Play or Understand ?", ("Play with it !", 
# 							        "Documentation")
# )
    
# st.write(f"Your choice : {result}")


    
def pick_random_class(class_names : list):
    
    target = random.choice(class_names)
    return (target)

def translate_class_names(name):
    
    french_name = GoogleTranslator(source='auto', target='fr').translate(name) 
    return (french_name)

class_names = load_class_names()   
tester_tab, doc_tab = st.tabs(["Play it !", "Documentation"])

with tester_tab :
    
    
    with st.form('input'):
        submit_button = st.form_submit_button(label='Get Word')

        if submit_button:
            target_class = pick_random_class(class_names)
            
            french_name = translate_class_names(target_class)
            
            st.subheader(f"- 🇬🇧 {target_class}")
            st.subheader(f"- 🇨🇵 {french_name}")
    
    
    canvas_result = st_canvas(stroke_width=2,
                        update_streamlit=True,
                        key="test")


    col1, col2, col3 = st.columns(3)
    
    if col2.button('Predict !'):

        draw = canvas_result.image_data
        if  (draw is not None):
            
            st.write(draw.shape)
            # arr_reshaped = draw.reshape(draw.shape[0], -1)
  
            # np.savetxt("merde.txt", arr_reshaped)
            # img = preprocessing_image(draw)
            # data = tf.expand_dims(img, axis=0)
            
            # st.image(img)
            # st.write(img.shape)
            
            # y_probability = model.predict(data)
            # argmax_index = y_probability.argmax()
            # y_pred = class_names[argmax_index]
            # # y_true = class_names[int(y_test[index])]
            # st.write(f"{y_pred}")

    
#     ### If prediction its good

# with doc_tab:
#     col1, col2, col3 = st.columns(3)
    
#     if col2.button('Click'):
#         st.write("salut")

        
    
    
    

    