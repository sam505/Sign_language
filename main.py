import tflite_runtime.interpreter as tflite
import streamlit as st
import numpy as np
import cv2
import os

interpreter = tflite.Interpreter("model/model.tflite")
labels = ['nothing', 'X', 'A', 'K', 'Z', 'V', 'space', 'I', 'O', 'S', 'del', 'L', 'C', 'N', 'G', 'Q', 'M', 'H', 'F',
          'E', 'Y', 'P', 'R', 'D', 'W', 'B', 'J', 'T', 'U']


def main():
    st.set_page_config(page_title='ASL Alphabets Predictor', page_icon=None, layout='centered',
                       initial_sidebar_state='auto')
    st.title("American Sign Language Alphabets Predictor")
    image = st.file_uploader(label="", type=['png', 'jpg', 'jpeg'])
    button = st.button("Upload Image")
    if button and image:
        save_uploaded_file(image)
        img_dir = "tempDir/" + image.name
        img = cv2.imread(img_dir)
        img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_disp)
        result = predict(img)
        label = np.argmax(result)
        st.text(result)
        st.text(labels[label])
        os.remove(img_dir)


def pre_process(image):
    image = cv2.resize(image, (100, 100))
    image = image.reshape(1, *image.shape)
    image = np.array(image, dtype='float32')

    return image


def predict(img):
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    img = pre_process(img)
    input_data = img
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def save_uploaded_file(uploaded_file):
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return None


if __name__ == '__main__':
    main()
