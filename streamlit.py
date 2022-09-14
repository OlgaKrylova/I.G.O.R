# Based
import pandas as pd
# Visualization
import seaborn as sns
# Web apps
import streamlit as st
# Stream
import cv2
# Live ML
import mediapipe as mp
# Models
import pickle
# Time
from datetime import datetime
# Music
from pygame import mixer
import pygame as pg
# Other functions
from main_functions import EyeClassifier, StaticPose, create_sleep_features_dict, create_features_dict, create_pose_features_dict, stat_vis
# Загружаем звук будильника
mixer.init()
mixer.music.load('songs/Song (mp3cut.net).mp3')
# PyGame2
pg.init()
sound1 = pg.mixer.Sound('songs/IGOR.wav')
sound2 = pg.mixer.Sound('songs/IGOR2.wav')
sound3 = pg.mixer.Sound('songs/Song2.wav')

# Загружаем модели детекции сна и эмоций по позе и лицу
face_cb_model = pickle.load(open('ML_models/cl_face_cb.sav', 'rb'))
pose_cb_model = pickle.load(open('ML_models/cl_pose_cb.sav', 'rb'))
emo_cb_model = pickle.load(open('ML_models/cl_emo_cb.sav', 'rb'))
# Загружаем инструменты из библиотеки mediapipe
mp_face_mesh = mp.solutions.face_mesh # подключаем инструменты для рисования сетки
mp_face_detection = mp.solutions.face_detection # подключаем инструменты для детекции
mp_drawing = mp.solutions.drawing_utils # подключаем инструменты для рисования
mp_drawing_styles = mp.solutions.drawing_styles # подключаем стили
mp_pose = mp.solutions.pose # распознавание поз
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Главная страница
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col3:
    st.header('I.')
with col4:
    st.header('G.')
with col5:
    st.header('O.')
with col6:
    st.header('R.')
add_radio = st.selectbox(" ",("Stream", "Statistics"))

# Считываем видеопоток
cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
# Создаем счётчик, списки и словари и пустые датасеты
points = [i for i in range (469)]

points_dict = {i : 0 for i in range(469)}

face_data = {
'r_shoulder_lip': 0,'r_shoulder_cheek': 0,'r_shoulder_eye': 0,'r_shoulder_eye_h': 0,'r_shoulder_eye_w': 0,
'l_shoulder_lip': 0,'l_shoulder_cheek': 0,'l_shoulder_eye': 0,'l_shoulder_eye_h': 0,'l_shoulder_eye_w': 0
            }

stat_dict_empt = {
    'Time': 0,'face_cb_pred': 0, 'EyeClassifier': 0, 'pose_cb_pred': 0, 'static_pose': 0,\
    'sleep_predict': 0, 'cheater': 0, 'emotion': 0,'emo_sth': 0
                 }

teta_dump = {
    'teta1':0,'teta2':0,'teta3': 0,'teta4': 0,'teta5': 0,'teta6': 0,'teta7': 0,'teta8': 0,'teta9': 0,'teta10': 0
            }

P = {
    0:	61, 1: 292, 2: 0, 3: 17, 4:	50,	5: 280,	6: 48, 7: 4, 8:	289, 9:	206, 10: 426, 11: 133, 12: 130, 13: 159,\
14:	145, 15: 362, 16: 359, 17: 386, 18:	374, 19: 122, 20: 351, 21: 46, 22: 105, 23: 107, 24: 276, 25: 334, 26:	336
    }  

emo_dump = pd.DataFrame(teta_dump, index=[0])
stat_frame_empt = pd.DataFrame(stat_dict_empt, index = [0])
face_dump = pd.DataFrame(face_data, index=[0])
face_pred,eyes_pred,pose_pred,emo_list = [],[],[],[]
fps_counter = 0

# Создаем объекты классов из библиотеки mediapipe
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=50, refine_landmarks= True, min_detection_confidence=0.3, min_tracking_confidence=0.05)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
pose = mp_pose.Pose(model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Работа с видеопотоком
if add_radio == "Stream":
    if st.checkbox('Well met! Click here! 🖥'):
        st.title("HI, MY NAME IS I.G.O.R AND I'LL BE WATCHING YOU 👁‍🗨")
        sound1.play()
    run = st.checkbox('Turn on your webcam 🎥')
    while run:
        _, image = cap.read()
        if _:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)
        results1 = face_mesh.process(image)
        fps_counter += 1
        df ={}
        if results1.multi_face_landmarks:
          for face_landmarks in results1.multi_face_landmarks:
            for id, point in enumerate(face_landmarks.landmark):
                if id in points:
                    width, height, color = image.shape
                    width, height = int(point.x * height), int(point.y * width)
                    points_dict[id] = [width, height]
            # Находим расстояни между точками на лице
            data_frame = pd.DataFrame(create_sleep_features_dict(points_dict),index=[0])
            face_cb_pred = face_cb_model.predict(data_frame)
            face_pred.append(face_cb_pred)
            eyes_pred.append(EyeClassifier(points_dict))
            text4 = 'eyes_pred' + str(EyeClassifier(points_dict)) 
            text = 'face_cb_pred' + str(face_cb_pred) 
            df_emo = create_features_dict(image, points, P, face_mesh, s=0)
            emo_data_frame = pd.DataFrame(df_emo,index=[0])
            emo_dump = pd.concat([emo_dump, emo_data_frame])

            emo_pred = emo_cb_model.predict(emo_data_frame)
            emo_pred = emo_pred[0]
        
        else:
            face_cb_pred = 2
            face_pred.append(face_cb_pred)
            text = 'face_no_cb_pred'
            emo_pred = ['NO FACE']

        emo_list.append(emo_pred[0])
        
        # Находим расстояния между точками на теле
        results3 = pose.process(image)
        df = {}
        if results3.pose_landmarks:
            df = create_pose_features_dict(image, results3, df)
            data_frame = pd.DataFrame(df,index=[0])
            face_dump = pd.concat([face_dump, data_frame])
            pose_cb_pred = pose_cb_model.predict(data_frame)
            text1 = 'pose_cb_pred' + str(pose_cb_pred) 
            pose_pred.append(pose_cb_pred)
        else:
            pose_cb_pred = 2
            pose_pred.append(pose_cb_pred)

        if fps_counter % 20 == 0:
            # Собираем датасет для статистики
            statistic_dict= {}
            time_now = str(datetime.today())[5:-7]
            try:
                statistic_dict['Time'], statistic_dict['face_cb_pred'],statistic_dict['EyeClassifier'],statistic_dict['pose_cb_pred'],statistic_dict['static_pose'],\
                statistic_dict['sleep_predict'], statistic_dict['cheater'],statistic_dict['emotion'],statistic_dict['emo_sth'] = str(datetime.today())[5:-7], face_cb_pred, EyeClassifier(points_dict), pose_cb_pred, str(StaticPose(face_dump, 5, 50)), sleep, cheater, emo_pred, pd.Series(emo_list)[-10:].mode()[0]
            except:
                statistic_dict['Time'], statistic_dict['face_cb_pred'],statistic_dict['EyeClassifier'],statistic_dict['pose_cb_pred'],statistic_dict['static_pose'],\
                statistic_dict['sleep_predict'], statistic_dict['cheater'], statistic_dict['emotion'],statistic_dict['emo_sth'] = str(datetime.today())[5:-7], face_cb_pred, EyeClassifier(points_dict), pose_cb_pred, str(StaticPose(face_dump, 5, 50)), 0,0, emo_pred, pd.Series(emo_list)[-10:].mode()[0]
             
            statistic_frame = pd.DataFrame(statistic_dict,index=[0])
            stat_frame_empt = pd.concat([stat_frame_empt, statistic_frame])
            stat_frame_empt.to_csv('Frames_archive/stat_frame.csv')

        if fps_counter % 90 == 0:
            elements_eyes = eyes_pred[-5:]
            elements_face = face_pred[-5:]
            elements_pose = pose_pred[-5:]

            if (elements_face.count(elements_face[0]) == len(elements_face)) and (elements_face[0] == 1) or \
                (elements_eyes.count(elements_eyes[0]) == len(elements_eyes)) and (elements_eyes[0] == 1):
                statistic_dict= {}
                time_now = str(datetime.today())[5:-7]
                sleep =1
                statistic_dict['Time'], statistic_dict['face_cb_pred'],statistic_dict['EyeClassifier'],statistic_dict['pose_cb_pred'],statistic_dict['static_pose'],\
                statistic_dict['sleep_predict'], statistic_dict['cheater'],statistic_dict['emotion'],statistic_dict['emo_sth'] = str(datetime.today())[5:-7], face_cb_pred, EyeClassifier(points_dict), pose_cb_pred, str(StaticPose(face_dump, 5, 50)), sleep, 0, emo_pred, pd.Series(emo_list)[-10:].mode()[0]
                mixer.music.play()
            else:
                if face_cb_pred == 2 and StaticPose(face_dump, 5, 50) == True and \
                    (elements_pose.count(elements_pose[0]) == len(elements_pose)) and (elements_pose[0] == 1):
                    statistic_dict= {}
                    time_now = str(datetime.today())[5:-7]
                    sleep=1
                    statistic_dict['Time'], statistic_dict['face_cb_pred'],statistic_dict['EyeClassifier'],statistic_dict['pose_cb_pred'],statistic_dict['static_pose'],\
                    statistic_dict['sleep_predict'], statistic_dict['cheater'], statistic_dict['emotion'],statistic_dict['emo_sth'] = str(datetime.today())[5:-7], face_cb_pred, EyeClassifier(points_dict), pose_cb_pred, str(StaticPose(face_dump, 5, 50)), sleep, 0, emo_pred, pd.Series(emo_list)[-10:].mode()[0]
                    mixer.music.play()

                elif ((elements_face.count(elements_face[0]) == len(elements_face) and elements_face[0] == 0) or \
                    (elements_eyes.count(elements_eyes[0]) == len(elements_eyes) and elements_eyes[0] == 0)) and \
                    StaticPose(face_dump, 50, 10) == True:
                    cheater =1
                    sound3.play()
                    statistic_dict['Time'], statistic_dict['face_cb_pred'],statistic_dict['EyeClassifier'],statistic_dict['pose_cb_pred'],statistic_dict['static_pose'],\
                    statistic_dict['sleep_predict'], statistic_dict['cheater'], statistic_dict['emotion'],statistic_dict['emo_sth'] = str(datetime.today())[5:-7], face_cb_pred, EyeClassifier(points_dict), pose_cb_pred, str(StaticPose(face_dump, 5, 50)), 0, cheater, emo_pred, pd.Series(emo_list)[-10:].mode()[0]
                else: 
                    sleep  = 0
                    statistic_dict['Time'], statistic_dict['face_cb_pred'],statistic_dict['EyeClassifier'],statistic_dict['pose_cb_pred'],statistic_dict['static_pose'],\
                    statistic_dict['sleep_predict'], statistic_dict['cheater'], statistic_dict['emotion'],statistic_dict['emo_sth'] = str(datetime.today())[5:-7], face_cb_pred, EyeClassifier(points_dict), pose_cb_pred, str(StaticPose(face_dump, 5, 50)), sleep, 0,emo_pred, pd.Series(emo_list)[-10:].mode()[0]
            statistic_frame = pd.DataFrame(statistic_dict,index=[0])
            stat_frame_empt = pd.concat([stat_frame_empt, statistic_frame])
            stat_frame_empt.to_csv('Frames_archive/stat_frame.csv')
                       
    else:
        st.write('')  
    cv2.destroyAllWindows()
    cap.release()

elif add_radio == "Statistics":
    choice = st.button('Visualization for every 10 minuts 📈')
    choice2 = st.button('Visualization for every hours 📉')
    choice3 = st.button('Table data visualization  📊')

    # Обрабатываем датафрейм для статистики
    stat_frame = pd.read_csv('Frames_archive/stat_frame.csv')
    stat_frame.drop(['Unnamed: 0','static_pose', 'emotion','face_cb_pred','EyeClassifier','pose_cb_pred'],axis=1,inplace=True)
    stat_frame = stat_frame.rename(columns={'emo_sth':'emotion'})
    stat_frame = stat_frame.iloc[1:]
    stat_frame = pd.get_dummies(data = stat_frame, columns = ['emotion'])
    stat_frame['Hour'] = [val[6:8] for val in stat_frame['Time']]
    stat_frame['decade'] = [val[9:10] for val in stat_frame['Time']]
    stat_frame['Date'] = [val[0:5] for val in stat_frame['Time']]

    # Параметры визуализации
    sns.set(style="white", palette="muted", color_codes=True)

    if choice:
        st.title('YOU DID NOT BELIEVE?')
        st.write('Total for every 10 minutes')
        sound2.play()
        # Визуализация за каждые 10 минут
        df = stat_frame.groupby(by='decade').sum()
        df.rename(index={i : i+str(0)+' to '+str(int(i)+1)+str(0) +' min' for i in list(df.index)},inplace=True)
        stat_vis(df)

    if choice2:
        st.title('YOU DID NOT BELIEVE?')
        st.write('Total for every hour')
        sound2.play()
        # Визуализация за каждый час
        df = stat_frame.groupby(by='Hour').sum()
        df.rename(index={i : i + ' hour' for i in list(df.index)},inplace=True)
        stat_vis(df)

    if choice3:
        st.title('YOU DID NOT BELIEVE?')
        sound2.play()
        # Визуализация датафрейма
        stat_frame.drop(['Hour','decade','Date'],axis=1,inplace=True)
        st.dataframe(stat_frame)