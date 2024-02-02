# image_classifier.py
import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
from PIL import ImageOps  # Install pillow instead of PIL
import numpy as np
from dotenv import load_dotenv 
import os
import image_classifier
import chatbot
import cv2
PAGES = {
    "이미지 판별": image_classifier,
    "챗봇": chatbot
   
}
def classify_waste(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r",encoding="utf-8").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    return class_name, confidence_score

def app():
    st.title("장애인 확인 등록시스템")
    option = st.selectbox("옵션을 선택하세요:", ["이미지 업로드", "웹캠 사용"])
    if option == "이미지 업로드":
        input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

        if input_img is not None:
            if st.button("확인"):
                image_file = Image.open(input_img)
                label, confidence_score = classify_waste(image_file)
                col1, col2 = st.columns([1,1])

                with col1:
                    st.info("업로드 하신 이미지")
                    st.image(input_img, use_column_width=True)
                
                with col2:
                    st.info("결과")
                    image_file = Image.open(input_img)
                    label, confidence_score = classify_waste(image_file)
                    if label.strip() == "0 장애인입니다.":
                        st.success("장애인등록 확인이 완료되었습니다.")
                        st.image("sdg goals/pngegg.png", use_column_width=True)
                        ## 여기가 페이지 이동 시킬 코드가 있어야 하는곳 
                        st.session_state.next_page = "카카오지도"
                    elif label.strip() == "1 비장애인":
                        st.success("장애인 등록 확인이 불가능합니다.")
                        st.image("sdg goals/pngegg (1).png", use_column_width=True)
                        st.session_state.next_page = "챗봇"
                    else:
                        st.error("사진을 다시 확인해주십시오.")
    elif option == "웹캠 사용":
        
        # 웹캠에서 비디오를 캡처하고 프레임을 분류하기 위한 코드를 추가합니다.
        cap = cv2.VideoCapture(0)

        if st.button("웹캠 시작"):
            st.write("웹캠 사용 중")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # OpenCV 프레임을 PIL 이미지로 변환합니다.
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 현재 프레임을 Streamlit에서 표시합니다.
                st.image(pil_frame, use_column_width=True)

                # 프레임을 분류합니다 (classify_waste 함수를 사용할 수 있습니다).
                x, y = classify_waste(pil_frame)
                co1, co2 = st.columns([1,1])

                with co1:
                    st.info("업로드 하신 이미지")
                    st.image(pil_frame, use_column_width=True)
                
                with co2:
                    st.info("결과")
    
                    x, y = classify_waste(pil_frame)
                    if x.strip() == "0 장애인입니다.":
                        st.success("장애인등록 확인이 완료되었습니다.")
                        st.image("sdg goals/pngegg.png", use_column_width=True)
                        ## 여기가 페이지 이동 시킬 코드가 있어야 하는곳 
                        st.session_state.next_page = "챗봇"
                    elif x.strip() == "1 비장애인":
                        st.success("장애인 등록 확인이 불가능합니다.")
                        st.image("sdg goals/pngegg (1).png", use_column_width=True)
                    else:
                        st.error("사진을 다시 확인해주십시오.")
                if st.button("중지"):
                    break

        # 사용이 끝나면 웹캠을 해제합니다.
        cap.release()
