import cv2
import mediapipe as mp
import datetime
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from pygame import mixer
import numpy as np
import math as m
from main_functions import euc_distance, create_features_dict, EyeClassifier, StaticPose

# загружаем звук будильника
mixer.init()
mixer.music.load('/Users/olgakrylova/ds_bootcamp/Emotions/budilnik.wav')
# загружаем модель
class_labels = ['Happy', 'Sad', 'Neutral', 'Angry']
emo_cb_model = pickle.load(open('ML_models/cl_emo_cb.sav', 'rb'))

face_data = {'teta1':0,'teta2':0,'teta3': 0,'teta4': 0,'teta5': 0,'teta6': 0,'teta7': 0,
'teta8': 0,'teta9': 0,'teta10': 0}
emo_dump = pd.DataFrame(face_data, index=[0])
            
mp_face_mesh = mp.solutions.face_mesh # подключаем инструменты для рисования сетки
mp_face_detection = mp.solutions.face_detection # подключаем инструменты для детекции
mp_drawing = mp.solutions.drawing_utils # подключаем инструменты для рисования
mp_drawing_styles = mp.solutions.drawing_styles # подключаем стили
mp_pose = mp.solutions.pose #распознавание поз

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

points = [i for i in range (469)]
P = {0:	61, 1: 292, 2: 0, 3: 17, 4:	50,	5: 280,	6: 48, 7: 4, 8:	289, 9:	206, 10: 426, 11: 133, 12: 130, 13: 159,\
14:	145, 15: 362, 16: 359, 17: 386, 18:	374, 19: 122, 20: 351, 21: 46, 22: 105, 23: 107, 24: 276, 25: 334, 26:	336}

face_pred = []
pose_pred = []
eyes_pred = []
full_pred =[]
text = 'none'
i = 0

face_mesh = mp_face_mesh.FaceMesh(
  max_num_faces=50,
  refine_landmarks= True,
  min_detection_confidence=0.3,
  min_tracking_confidence=0.05)

face_detection = mp_face_detection.FaceDetection(
model_selection=1, min_detection_confidence=0.3)

pose = mp_pose.Pose(model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

####### РАБОТА С ВИДЕОПОТОКОМ
while (True):
    ret, image = cap.read(0)   
    i += 1
    results1 = face_mesh.process(image)
    if results1.multi_face_landmarks:
      # отрисовываем точки на изображении
      for face_landmarks in results1.multi_face_landmarks:
        mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style()) 
    df = create_features_dict(image, points, P, face_mesh, s=0)
    emo_data_frame = pd.DataFrame(df,index=[0])
    emo_dump = pd.concat([emo_dump, emo_data_frame])
    print(emo_data_frame.columns)
    emo_pred = emo_cb_model.predict(emo_data_frame)
    text = 'emo_pred' + str(emo_pred) 

    ### ВЫВОД НА ЭКРАН
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(image, text, org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('results', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()