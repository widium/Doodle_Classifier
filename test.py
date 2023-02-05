
import gradio as gr
from gradio import Interface
from gradio import Textbox
from gradio import TextArea
from gradio import Sketchpad
from gradio import Label

from keras.models import load_model
from PIL.Image import fromarray
from PIL import Image

import cv2
import numpy as np
import tensorflow as tf


model = load_model("best_model")


def create_probability_class_dict(y_probability, class_names):
    
    probabilitys = dict()

    for index, value in enumerate(y_probability[0]):
        probabilitys[class_names[index]] = float(value)
    return (probabilitys)

    # for index, value in probability_dict.items():
    #     print(type(value))

def load_class_names():
    my_file = open("class_names.txt", "r")
    data = my_file.read()
    class_names = data.split("\n")
    my_file.close()
    return (class_names)
    

def preprocessing_image(draw):
    
    draw = draw/255.
    draw = tf.reshape(draw, (1, draw.shape[0], draw.shape[1], 1))
    return (draw)

def predict(draw):
    class_names = load_class_names()
    img = preprocessing_image(draw)
    y_probability = model.predict(img)
    probability_dict = create_probability_class_dict(y_probability, class_names)
    return (probability_dict)
    # argmax_index = y_probability.argmax()
    # y_pred = class_names[argmax_index]
    # y_true = class_names[int(y_test[index])]

# .style(height=400, width=400)

# text_box = Textbox(label="Name Recognition",
#                     lines=3,
#                     placeholder="Name Here")

title = "Welcome on Deep Pictionary ✏ 🧠"

head = (
  "<center>"
  "<img src='cookie.png', width=400>"
  "the AI was trained on 100 class"
  "</center>"
)

ref = "Find the whole code [here](https://github.com/ovh/ai-training-examples/tree/main/apps/gradio/sketch-recognition)."

draw_box = Sketchpad(shape=(28, 28))
label_box = Label(num_top_classes=10)

demo = gr.Interface(fn=predict, 
                    inputs=draw_box, 
                    outputs=label_box,
                    live=True,
                    title=title,
                    description=head,
                    article=ref)

demo.launch()   