import streamlit  as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 

MODEL_PATH = "mybatch_students_cnn_classifier.h5"

model = load_model(MODEL_PATH)

class_names = class_names = ['anu','bharti','deepak','sudh']

st.set_page_config(page_title = "my student image class", layout = 'centered' )

st.sidebar.title("upload your image")
st.markdown("this application will try to give you a classification of your image its based on vanila CNN architecture")

upload_file = st.sidebar.file_uploader("choose your image ", type = ["jpg", "jpeg", "png"])

from PIL import Image

if upload_file is not None :
    img = Image.open(upload_file).convert('RGB')
    st.image(img, caption = "your image")
    
    image_resized = img.resize((128,128))
    image_array = image.img_to_array(image_resized)/255.0
    image_batch = np.expand_dims(image_array,axis = 0)
    
    prediction = model.predict(image_batch)
    predicted_class = class_names[np.argmax(prediction)]
    
    st.success(f"this image is predicted to be : {predicted_class}")
    
    st.subheader("below is your confidence score for all the class")
    print(prediction)
    
    for index , score in enumerate(prediction[0]):
        st.write(f"{class_names[index]} : {score}")
    
    
    