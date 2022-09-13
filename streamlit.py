# based
import pandas as pd
import numpy as np
import math as m
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# web apps
import streamlit as st
#stream
import cv2
# live ML
import mediapipe as mp
#models
import pickle
# time
from datetime import datetime, date, time
# music
from pygame import mixer
import pygame as pg
# other functions
from main_functions import euc_distance, EyeClassifier, StaticPose, create_sleep_features_dict, create_features_dict, create_pose_features_dict

# загружаем звук будильника
mixer.init()
mixer.music.load('songs/Song (mp3cut.net).mp3')
#PyGame2
pg.init()
sound1 = pg.mixer.Sound('songs/IGOR.wav')
sound2 = pg.mixer.Sound('songs/IGOR2.wav')
sound3 = pg.mixer.Sound('songs/Song2.wav')

# загружаем модели детекции сна и эмоций по позе и лицу
face_cb_model = pickle.load(open('ML_models/cl_face_cb.sav', 'rb'))
pose_cb_model = pickle.load(open('ML_models/cl_pose_cb.sav', 'rb'))
emo_cb_model = pickle.load(open('ML_models/cl_emo_cb.sav', 'rb'))
# загружаем инструменты из библиотеки mediapipe
mp_face_mesh = mp.solutions.face_mesh # подключаем инструменты для рисования сетки
mp_face_detection = mp.solutions.face_detection # подключаем инструменты для детекции
mp_drawing = mp.solutions.drawing_utils # подключаем инструменты для рисования
mp_drawing_styles = mp.solutions.drawing_styles # подключаем стили
mp_pose = mp.solutions.pose # распознавание поз
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#IGOR in streamlit, OMG!
st.title('Select Format')
add_radio = st.selectbox(" ",("Stream", "Statistics"))
FRAME_WINDOW = st.image([])
if st.button('🖥'):
    st.title("HI, MY NAME IS I.G.O.R AND I'LL BE WATCHING YOU 👁‍🗨")
    sound1.play()

# считываем видеопоток
cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
# создаем счётчик, списки и словари и пустые датасеты
points = [i for i in range (469)]
points_dict = {i : 0 for i in range(469)}

face_data = {
'r_shoulder_lip': 0,'r_shoulder_cheek': 0,'r_shoulder_eye': 0,'r_shoulder_eye_h': 0,'r_shoulder_eye_w': 0,
'l_shoulder_lip': 0,'l_shoulder_cheek': 0,'l_shoulder_eye': 0,'l_shoulder_eye_h': 0,'l_shoulder_eye_w': 0
            }

stat_dict_empt = {'Time': 0,'face_cb_pred': 0, 'EyeClassifier': 0, 'pose_cb_pred': 0, 'static_pose': 0,'sleep_predict': 0, 'cheater': 0, 'emotion': 0,'emo_sth': 0}

teta_dump = {'teta1':0,'teta2':0,'teta3': 0,'teta4': 0,'teta5': 0,'teta6': 0,'teta7': 0,
'teta8': 0,'teta9': 0,'teta10': 0}

P = {
    0:	61, 1: 292, 2: 0, 3: 17, 4:	50,	5: 280,	6: 48, 7: 4, 8:	289, 9:	206, 10: 426, 11: 133, 12: 130, 13: 159,\
14:	145, 15: 362, 16: 359, 17: 386, 18:	374, 19: 122, 20: 351, 21: 46, 22: 105, 23: 107, 24: 276, 25: 334, 26:	336
    }  

emo_dump = pd.DataFrame(teta_dump, index=[0])
stat_frame_empt = pd.DataFrame(stat_dict_empt, index = [0])
face_dump = pd.DataFrame(face_data, index=[0])

face_pred,eyes_pred,pose_pred,emo_list = [],[],[],[]
text4,text,text1,text2 = '','','',''

fps_counter = 0

# создаем объекты классов из библиотеки mediapipe
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=50, refine_landmarks= True, min_detection_confidence=0.3, min_tracking_confidence=0.05)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
pose = mp_pose.Pose(model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

####### РАБОТА С ВИДЕОПОТОКОМ
if add_radio == "Stream":
    run = st.checkbox('🎥')
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
            # находим расстояни между точками на лице
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
        
        # находим расстояния между точками на теле
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
            # собираем датасет для статистики
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
    choice = st.checkbox('Hour plot 📈')
    choice2 = st.checkbox('Day plot 📉')
    choice3 = st.checkbox('Table data 📊')
    stat_frame = pd.read_csv('Frames_archive/stat_frame.csv')
    stat_frame.drop(['Unnamed: 0','static_pose'],axis=1,inplace=True)
    stat_frame = stat_frame.iloc[1:]
    st.title('YOU DID NOT BELIEVE?')
    sound2.play()
    if choice:
        # # обработка датафрейма
        stat_frame = pd.read_csv('Frames_archive/stat_frame.csv')
        stat_frame.drop(['Unnamed: 0','static_pose'],axis=1,inplace=True)
        stat_frame = stat_frame.iloc[1:]
        stat1 = stat_frame.copy()
        stat1 = pd.get_dummies(data = stat1, columns = ['emotion'])
        stat1['Hour'] = [val[6:8] for val in stat1['Time']]
        stat1['decade'] = [val[9:10] for val in stat1['Time']]
        stat1['Date'] = [val[0:5] for val in stat1['Time']]
        stat1.drop('Time',axis=1,inplace=True)
        df = stat1.groupby(by='decade').sum()
        df.rename(index={i : i+str(0)+' to '+str(int(i)+1)+str(0) for i in list(df.index)},inplace=True)
        df.rename(columns={'sleep_predict':'Sleep','cheater': 'Cheating detected','emotion_NO FACE': 'No face detected', 'emotion_disgust': 'Disgust','emotion_happy': 'Happy','emotion_neutral': 'Neutral','emotion_sad': 'Sad','emotion_surprise': 'Surprise'},inplace=True)
        # Визуализация

        fig, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8, 1, figsize=(8, 10), sharex=True)
        x = np.array(list(df.index))

        try:
            y1 = df['Neutral'].values
            ax1 = sns.barplot(x=x, y=y1, palette="ch:s=.25,rot=-.25", ax=ax1)
            for bar in ax1.patches:
                ax1.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax1.axhline(0, color="k", clip_on=False)
            ax1.set_ylabel("neutral")
        except:
            df['Neutral'] = 0
            y1 = df['Neutral'].values
            ax1 = sns.barplot(x=x, y=y1, palette="ch:s=.25,rot=-.25", ax=ax1)
            for bar in ax1.patches:
                ax1.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax1.axhline(0, color="k", clip_on=False)
            ax1.set_ylabel("neutral")

        try:
            y2 = df['Happy'].values
            ax2 = sns.barplot(x=x, y=y2, palette="light:#5A9", ax=ax2)
            for bar in ax2.patches:
                ax2.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax2.axhline(0, color="k", clip_on=False)
            ax2.set_ylabel("happy")
        except:
            df['Happy'] = 0
            y2 = df['Happy'].values
            ax2 = sns.barplot(x=x, y=y2, palette="light:#5A9", ax=ax2)
            for bar in ax2.patches:
                ax2.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax2.axhline(0, color="k", clip_on=False)
            ax2.set_ylabel("happy")

        try:
            y3 = df['Sad'].values
            ax3 = sns.barplot(x=x, y=y3, palette="ch:s=.25,rot=-.25", ax=ax3)
            for bar in ax3.patches:
                ax3.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax3.axhline(0, color="k", clip_on=False)
            ax3.set_ylabel("sad")
        except:
            df['Sad'] = 0
            y3 = df['Sad'].values
            ax3 = sns.barplot(x=x, y=y3, palette="ch:s=.25,rot=-.25", ax=ax3)
            for bar in ax3.patches:
                ax3.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax3.axhline(0, color="k", clip_on=False)
            ax3.set_ylabel("sad")

        try:
            y4 = df['Surprise'].values
            ax4 = sns.barplot(x=x, y=y4, palette="light:#5A9", ax=ax4)
            for bar in ax4.patches:
                ax4.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax4.axhline(0, color="k", clip_on=False)
            ax4.set_ylabel("surprise")
        except:
            df['Surprise'] = 0
            y4 = df['Surprise'].values
            ax4 = sns.barplot(x=x, y=y4, palette="light:#5A9", ax=ax4)
            for bar in ax4.patches:
                ax4.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax4.axhline(0, color="k", clip_on=False)
            ax4.set_ylabel("surprise")

        try:
            y5 = df['Disgust'].values
            ax5 = sns.barplot(x=x, y=y5, palette="ch:s=.25,rot=-.25", ax=ax5)
            for bar in ax5.patches:
                ax5.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax5.axhline(0, color="k", clip_on=False)
            ax5.set_ylabel("disgust")
        except:
            df['Disgust'] = 0
            y5 = df['Disgust'].values
            ax5 = sns.barplot(x=x, y=y5, palette="ch:s=.25,rot=-.25", ax=ax5)
            for bar in ax5.patches:
                ax5.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax5.axhline(0, color="k", clip_on=False)
            ax5.set_ylabel("disgust")

        try:
            y6 = df['No face detected'].values
            ax6 = sns.barplot(x=x, y=y6, palette="light:#5A9", ax=ax6)
            for bar in ax6.patches:
                ax6.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax6.axhline(0, color="k", clip_on=False)
            ax6.set_ylabel("No face detected")
        except:
            df['No face detected'] = 0
            y6 = df['No face detected'].values
            ax6 = sns.barplot(x=x, y=y6, palette="light:#5A9", ax=ax6)
            for bar in ax6.patches:
                ax6.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax6.axhline(0, color="k", clip_on=False)
            ax6.set_ylabel("No face detected")
        try:
            y7 = df['Sleep'].values
            ax7 = sns.barplot(x=x, y=y7, palette="ch:s=.25,rot=-.25", ax=ax7)
            for bar in ax7.patches:
                ax7.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax7.axhline(0, color="k", clip_on=False)
            ax7.set_ylabel("Sleep")
        except:
            df['Sleep'] = 0
            y7 = df['Sleep'].values
            ax7 = sns.barplot(x=x, y=y7, palette="ch:s=.25,rot=-.25", ax=ax7)
            for bar in ax7.patches:
                ax7.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax7.axhline(0, color="k", clip_on=False)
            ax7.set_ylabel("Sleep")

        try:
            y8 = df['Cheating detected'].values
            ax8 = sns.barplot(x=x, y=y8, palette="light:#5A9", ax=ax8)
            for bar in ax8.patches:
                ax8.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax8.axhline(0, color="k", clip_on=False)
            ax8.set_ylabel("Cheating detected")
            ax8.set_xlabel('Total for every 10 minutes')
        except:
            df['Cheating detected'] = 0
            y8 = df['Cheating detected'].values
            ax8 = sns.barplot(x=x, y=y8, palette="light:#5A9", ax=ax8)
            for bar in ax8.patches:
                ax8.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax8.axhline(0, color="k", clip_on=False)
            ax8.set_ylabel("Cheating detected")
            ax8.set_xlabel('Total for every 10 minutes')

        # Finalize the plot
        sns.despine(bottom=True)
        plt.setp(fig.axes, yticks=[])
        plt.tight_layout(h_pad=2)
        st.pyplot(fig)
    if choice2:
        stat_frame = pd.read_csv('Frames_archive/stat_frame.csv')
        stat_frame.drop(['Unnamed: 0','static_pose'],axis=1,inplace=True)
        stat_frame = stat_frame.iloc[1:]
        stat1 = stat_frame.copy()
        stat1 = pd.get_dummies(data = stat1, columns = ['emotion'])
        stat1['Hour'] = [val[6:8] for val in stat1['Time']]
        stat1['decade'] = [val[9:10] for val in stat1['Time']]
        stat1['Date'] = [val[0:5] for val in stat1['Time']]
        stat1.drop('Time',axis=1,inplace=True)
        df = stat1.groupby(by='Hour').sum()
        df.rename(index={i : i + ' hour' for i in list(df.index)},inplace=True)
        df.rename(columns={'sleep_predict':'Sleep','cheater': 'Cheating detected','emotion_NO FACE': 'No face detected', 'emotion_disgust': 'Disgust','emotion_happy': 'Happy','emotion_neutral': 'Neutral','emotion_sad': 'Sad','emotion_surprise': 'Surprise'},inplace=True)
        # Визуализация

        fig, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8, 1, figsize=(8 , 12), sharex=True)

        x = np.array(list(df.index))
        try:
            y1 = df['Neutral'].values
            ax1 = sns.barplot(x=x, y=y1, palette="ch:s=.25,rot=-.25", ax=ax1)
            for bar in ax1.patches:
                ax1.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax1.axhline(0, color="k", clip_on=False)
            ax1.set_ylabel("neutral")
        except:
            df['Neutral'] = 0
            y1 = df['Neutral'].values
            ax1 = sns.barplot(x=x, y=y1, palette="ch:s=.25,rot=-.25", ax=ax1)
            for bar in ax1.patches:
                ax1.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax1.axhline(0, color="k", clip_on=False)
            ax1.set_ylabel("neutral")

        try:
            y2 = df['Happy'].values
            ax2 = sns.barplot(x=x, y=y2, palette="light:#5A9", ax=ax2)
            for bar in ax2.patches:
                ax2.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax2.axhline(0, color="k", clip_on=False)
            ax2.set_ylabel("happy")
        except:
            df['Happy'] = 0
            y2 = df['Happy'].values
            ax2 = sns.barplot(x=x, y=y2, palette="light:#5A9", ax=ax2)
            for bar in ax2.patches:
                ax2.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax2.axhline(0, color="k", clip_on=False)
            ax2.set_ylabel("happy")

        try:
            y3 = df['Sad'].values
            ax3 = sns.barplot(x=x, y=y3, palette="ch:s=.25,rot=-.25", ax=ax3)
            for bar in ax3.patches:
                ax3.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax3.axhline(0, color="k", clip_on=False)
            ax3.set_ylabel("sad")
        except:
            df['Sad'] = 0
            y3 = df['Sad'].values
            ax3 = sns.barplot(x=x, y=y3, palette="ch:s=.25,rot=-.25", ax=ax3)
            for bar in ax3.patches:
                ax3.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax3.axhline(0, color="k", clip_on=False)
            ax3.set_ylabel("sad")

        try:
            y4 = df['Surprise'].values
            ax4 = sns.barplot(x=x, y=y4, palette="light:#5A9", ax=ax4)
            for bar in ax4.patches:
                ax4.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax4.axhline(0, color="k", clip_on=False)
            ax4.set_ylabel("surprise")
        except:
            df['Surprise'] = 0
            y4 = df['Surprise'].values
            ax4 = sns.barplot(x=x, y=y4, palette="light:#5A9", ax=ax4)
            for bar in ax4.patches:
                ax4.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax4.axhline(0, color="k", clip_on=False)
            ax4.set_ylabel("surprise")

        try:
            y5 = df['Disgust'].values
            ax5 = sns.barplot(x=x, y=y5, palette="ch:s=.25,rot=-.25", ax=ax5)
            for bar in ax5.patches:
                ax5.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax5.axhline(0, color="k", clip_on=False)
            ax5.set_ylabel("disgust")
        except:
            df['Disgust'] = 0
            y5 = df['Disgust'].values
            ax5 = sns.barplot(x=x, y=y5, palette="ch:s=.25,rot=-.25", ax=ax5)
            for bar in ax5.patches:
                ax5.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax5.axhline(0, color="k", clip_on=False)
            ax5.set_ylabel("disgust")

        try:
            y6 = df['No face detected'].values
            ax6 = sns.barplot(x=x, y=y6, palette="light:#5A9", ax=ax6)
            for bar in ax6.patches:
                ax6.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax6.axhline(0, color="k", clip_on=False)
            ax6.set_ylabel("No face detected")
        except:
            df['No face detected'] = 0
            y6 = df['No face detected'].values
            ax6 = sns.barplot(x=x, y=y6, palette="light:#5A9", ax=ax6)
            for bar in ax6.patches:
                ax6.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax6.axhline(0, color="k", clip_on=False)
            ax6.set_ylabel("No face detected")
        try:
            y7 = df['Sleep'].values
            ax7 = sns.barplot(x=x, y=y7, palette="ch:s=.25,rot=-.25", ax=ax7)
            for bar in ax7.patches:
                ax7.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax7.axhline(0, color="k", clip_on=False)
            ax7.set_ylabel("Sleep")
        except:
            df['Sleep'] = 0
            y7 = df['Sleep'].values
            ax7 = sns.barplot(x=x, y=y7, palette="ch:s=.25,rot=-.25", ax=ax7)
            for bar in ax7.patches:
                ax7.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax7.axhline(0, color="k", clip_on=False)
            ax7.set_ylabel("Sleep")

        try:
            y8 = df['Cheating detected'].values
            ax8 = sns.barplot(x=x, y=y8, palette="light:#5A9", ax=ax8)
            for bar in ax8.patches:
                ax8.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax8.axhline(0, color="k", clip_on=False)
            ax8.set_ylabel("Cheating detected")
            ax8.set_xlabel('Total for every hour')
        except:
            df['Cheating detected'] = 0
            y8 = df['Cheating detected'].values
            ax8 = sns.barplot(x=x, y=y8, palette="light:#5A9", ax=ax8)
            for bar in ax8.patches:
                ax8.annotate(format(bar.get_height(), '.2f'),
                (bar.get_x() + bar.get_width() / 2,
                bar.get_height()), ha='center', va='center',
                size=10, xytext=(0, 5),
                textcoords='offset points')
            ax8.axhline(0, color="k", clip_on=False)
            ax8.set_ylabel("Cheating detected")
            ax8.set_xlabel('Total for every hour')

        # Finalize the plot
        sns.despine(bottom=True)
        plt.setp(fig.axes, yticks=[])
        plt.tight_layout(h_pad=2)
        st.pyplot(fig)

    if choice3:
         # # обработка датафрейма
        stat_frame = pd.read_csv('Frames_archive/stat_frame.csv')
        stat_frame.drop(['Unnamed: 0','static_pose'],axis=1,inplace=True)
        stat_frame = stat_frame.iloc[1:]
        stat1 = stat_frame.copy()
        stat1 = pd.get_dummies(data = stat1, columns = ['emotion'])
        stat1['Hour'] = [val[6:8] for val in stat1['Time']]
        stat1['decade'] = [val[9:10] for val in stat1['Time']]
        stat1['Date'] = [val[0:5] for val in stat1['Time']]
        stat1.drop('Time',axis=1,inplace=True)
        df = stat1.groupby(by='decade').sum()
        df.rename(index={i : i+str(0)+' to '+str(int(i)+1)+str(0) for i in list(df.index)},inplace=True)
        df.rename(columns={'sleep_predict':'Sleep','cheater': 'Cheating detected','emotion_NO FACE': 'No face detected', 'emotion_disgust': 'Disgust','emotion_happy': 'Happy','emotion_neutral': 'Neutral','emotion_sad': 'Sad','emotion_surprise': 'Surprise'},inplace=True)
        # Визуализация
        st.dataframe(df)