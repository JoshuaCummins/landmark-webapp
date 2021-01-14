import streamlit as st
import cv2
from PIL import Image
import numpy
import dlib


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(img):
	img = numpy.array(img)
	rects = detector(img, 1)

	if len(rects) > 1:
		raise TooManyFaces
	if len(rects) == 0:
		raise NoFaces

	return numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])

def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        
        cv2.circle(img, pos, 3, color=(32, 97, 41))
    return img



	


st.title("Facial Landmark generation")

html_temp = """
<body style="background-color:red;">
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Facial Landmark WebApp</h2>
</div>
</body>
"""
st.markdown(html_temp, unsafe_allow_html=True)

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
	img = Image.open(image_file)
	st.text("Original Image")
	st.image(img)

if st.button("Compute"):
	img = numpy.array(img)
	landmarks = get_landmarks(img)
	image_with_landmarks = annotate_landmarks(img, landmarks)
	st.image(image_with_landmarks)
	