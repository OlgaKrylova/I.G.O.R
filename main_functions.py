import math as m
import cv2
import mediapipe as mp
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def euc_distance(x,y):
  a = ((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5
  return a

def EyeClassifier(p_dict):
    predict = ((euc_distance(p_dict[362],p_dict[359])+0.0001)/(euc_distance(p_dict[374],p_dict[386])+0.0001) > 7)*\
    ((euc_distance(p_dict[133],p_dict[130])+0.0001)/(euc_distance(p_dict[159],p_dict[145])+0.0001) > 7)*1
    text = ['open', 'closed']
    return predict

def StaticPose (face_dump, points, num):  
    a=[]
    try:
      for i in range(points+1):
        if sum(abs(face_dump.iloc[-i-1] - face_dump.iloc[-i-2])) <= num:
          a.append(1)
        else:
          a.append(0)
      return (sum(a)/len(a)-0.98)>=0
    except:
      return ''

### EMO CLASSIFIER FEATURES CREATION 
def create_features_dict(image, points, P, face_mesh, s):
  df = {}
  points_dict = {i : 0 for i in range(469)}
  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      for id, point in enumerate(face_landmarks.landmark):
          if id in points:
              width, height, color = image.shape
              width, height = int(point.x * width), int(point.y * height)
              points_dict[id] = [width, height]

      ### ANGLES
      df['teta1'] = 57.296 *  m.acos(((points_dict[P[2]][0] - points_dict[P[0]][0])*(points_dict[P[3]][0] - points_dict[P[0]][0]) +\
      (points_dict[P[2]][1] - points_dict[P[0]][1])*(points_dict[P[3]][1] - points_dict[P[0]][1]))/\
      ((euc_distance(points_dict[P[0]],points_dict[P[2]]) * euc_distance(points_dict[P[0]],points_dict[P[3]]))+0.0001))
      teta2 = 57.296 * m.acos(((points_dict[P[0]][0] - points_dict[P[2]][0])*(points_dict[P[1]][0] - points_dict[P[2]][0])+\
            (points_dict[P[0]][1] - points_dict[P[2]][1])*(points_dict[P[1]][1] - points_dict[P[2]][1]))/\
          ((euc_distance(points_dict[P[0]],points_dict[P[2]]) * euc_distance(points_dict[P[2]],points_dict[P[1]]))+0.0001))
      sign = ((points_dict[P[1]][1] - points_dict[P[2]][1]<0) and (points_dict[P[0]][1] - points_dict[P[2]][1])<0) * (-1)
      df['teta2'] =m.copysign(teta2, sign)
      df['teta3'] = 57.296 *  m.acos(((points_dict[P[6]][0] - points_dict[P[7]][0])*(points_dict[P[8]][0] - points_dict[P[7]][0]) +\
          (points_dict[P[6]][1] - points_dict[P[7]][1])*(points_dict[P[8]][1] - points_dict[P[7]][1]))/ \
          ((euc_distance(points_dict[P[6]],points_dict[P[7]]) * euc_distance(points_dict[P[8]],points_dict[P[7]]))+0.0001) )
      df['teta4'] = 57.296 *  m.acos(((points_dict[P[9]][0] - points_dict[P[7]][0])*(points_dict[P[10]][0] - points_dict[P[7]][0]) +\
          (points_dict[P[9]][1] - points_dict[P[7]][1])*(points_dict[P[10]][1] - points_dict[P[7]][1]))/\
          ((euc_distance(points_dict[P[9]],points_dict[P[7]]) * euc_distance(points_dict[P[10]],points_dict[P[7]]))+0.0001))
      df['teta5'] = 57.296 *  m.acos(((points_dict[P[0]][0] - points_dict[P[7]][0])*(points_dict[P[1]][0] - points_dict[P[7]][0]) +\
          (points_dict[P[0]][1] - points_dict[P[7]][1])*(points_dict[P[1]][1] - points_dict[P[7]][1]))/\
          ((euc_distance(points_dict[P[0]],points_dict[P[7]]) * euc_distance(points_dict[P[1]],points_dict[P[7]]))+0.0001))
      df['teta6'] = 57.296 *  m.acos(((points_dict[P[1]][0] - points_dict[P[5]][0])*(points_dict[P[8]][0] - points_dict[P[5]][0]) +\
          (points_dict[P[1]][1] - points_dict[P[5]][1])*(points_dict[P[8]][1] - points_dict[P[5]][1]))/\
          ((euc_distance(points_dict[P[1]],points_dict[P[5]]) * euc_distance(points_dict[P[8]],points_dict[P[5]]))+0.0001))
      df['teta7'] = 57.296 *  m.acos(((points_dict[P[1]][0] - points_dict[P[10]][0])*(points_dict[P[8]][0] - points_dict[P[10]][0]) +\
          (points_dict[P[1]][1] - points_dict[P[10]][1])*(points_dict[P[8]][1] - points_dict[P[10]][1]))/\
          ((euc_distance(points_dict[P[1]],points_dict[P[10]]) * euc_distance(points_dict[P[8]],points_dict[P[10]]))+0.0001))
      df['teta8'] = 57.296 *  m.acos(((points_dict[P[13]][0] - points_dict[P[12]][0])*(points_dict[P[14]][0] - points_dict[P[12]][0]) +\
          (points_dict[P[13]][1] - points_dict[P[12]][1])*(points_dict[P[14]][1] - points_dict[P[12]][1]))/\
          ((euc_distance(points_dict[P[13]],points_dict[P[12]]) * euc_distance(points_dict[P[14]],points_dict[P[12]]))+0.0001) )
      df['teta9'] = 57.296 *  m.acos(((points_dict[P[21]][0] - points_dict[P[22]][0])*(points_dict[P[23]][0] - points_dict[P[22]][0]) +\
          (points_dict[P[21]][1] - points_dict[P[22]][1])*(points_dict[P[23]][1] - points_dict[P[22]][1]))/\
          ((euc_distance(points_dict[P[21]],points_dict[P[22]]) * euc_distance(points_dict[P[23]],points_dict[P[22]]))+0.0001))
      df['teta10'] = 57.296 *  m.acos(((points_dict[P[6]][0] - points_dict[P[19]][0])*(points_dict[P[23]][0] - points_dict[P[19]][0]) +\
          (points_dict[P[6]][1] - points_dict[P[19]][1])*(points_dict[P[23]][1] - points_dict[P[19]][1]))/\
          ((euc_distance(points_dict[P[6]],points_dict[P[19]]) * euc_distance(points_dict[P[23]],points_dict[P[19]]))+0.0001))

      ### DISTANCES
      df['l_eye_w'] = euc_distance(points_dict[362],points_dict[359])/euc_distance(points_dict[10],points_dict[152])
      df['l_eye_h'] = euc_distance(points_dict[386],points_dict[374])/euc_distance(points_dict[10],points_dict[152])
      df['r_eye_w'] = euc_distance(points_dict[130],points_dict[133])/euc_distance(points_dict[10],points_dict[152])
      df['r_eye_h'] = euc_distance(points_dict[159],points_dict[145])/euc_distance(points_dict[10],points_dict[152])
      df['lips_w'] = euc_distance(points_dict[61],points_dict[292])/euc_distance(points_dict[10],points_dict[152])
      df['lips_h'] = euc_distance(points_dict[0],points_dict[17])/euc_distance(points_dict[10],points_dict[152])
      df['lips_h_in'] = euc_distance(points_dict[13],points_dict[14])/euc_distance(points_dict[10],points_dict[152])
      df['brows_dist'] = euc_distance(points_dict[55],points_dict[285])/euc_distance(points_dict[10],points_dict[152])
      df['r_cheek_eye'] = euc_distance(points_dict[280],points_dict[446])/euc_distance(points_dict[10],points_dict[152])
      df['l_cheek_eye'] = euc_distance(points_dict[50],points_dict[226])/euc_distance(points_dict[10],points_dict[152])
      df['r_cheek_lip'] = euc_distance(points_dict[280],points_dict[287])/euc_distance(points_dict[10],points_dict[152])
      df['l_cheek_lip'] = euc_distance(points_dict[50],points_dict[57])/euc_distance(points_dict[10],points_dict[152])
      df['r_eye_brow_in'] = euc_distance(points_dict[285],points_dict[464])/euc_distance(points_dict[10],points_dict[152])
      df['l_eye_brow_in'] = euc_distance(points_dict[55],points_dict[243])/euc_distance(points_dict[10],points_dict[152])
      df['r_eye_brow_out'] = euc_distance(points_dict[300],points_dict[446])/euc_distance(points_dict[10],points_dict[152])
      df['l_eye_brow_out'] = euc_distance(points_dict[70],points_dict[226])/euc_distance(points_dict[10],points_dict[152])
      df['r_eye_nose_in'] = euc_distance(points_dict[133],points_dict[100])/euc_distance(points_dict[10],points_dict[152])
      df['l_eye_nose_in'] = euc_distance(points_dict[463],points_dict[329])/euc_distance(points_dict[10],points_dict[152])
  else:
      df = dict.fromkeys(['teta1', 'teta2', 'teta3', 'teta4', 'teta5', 'teta6', 'teta7', 'teta8', 'teta9', 'teta10', 'l_eye_w',\
              'l_eye_h', 'r_eye_w', 'r_eye_h', 'lips_w', 'lips_h', 'lips_h_in', 'brows_dist', 'r_cheek_eye', 'l_cheek_eye',\
              'r_cheek_lip', 'l_cheek_lip', 'r_eye_brow_in', 'l_eye_brow_in', 'r_eye_brow_out', 'l_eye_brow_out', 'r_eye_nose_in',\
                'l_eye_nose_in'],0)
  return df

def create_sleep_features_dict(points_dict):
  df = {}
  df['l_eye_w'] = euc_distance(points_dict[362],points_dict[359])
  df['l_eye_h'] = euc_distance(points_dict[386],points_dict[374])
  df['r_eye_w'] = euc_distance(points_dict[130],points_dict[133])
  df['r_eye_h'] = euc_distance(points_dict[159],points_dict[145])
  df['eyes_h_dist'] = abs(points_dict[159][1] - points_dict[386][1])
  df['eyes_w_dist'] = abs(points_dict[159][0] - points_dict[386][0])
  df['lips_h_dist'] = abs(points_dict[61][1] - points_dict[292][1])
  df['lips_w_dist'] = abs(points_dict[61][0] - points_dict[292][0])
  df['cheeks_h_dist'] = abs(points_dict[50][1] - points_dict[280][1])
  df['cheeks_w_dist'] = abs(points_dict[50][0] - points_dict[280][0])
  df['face_h_dist'] = abs(points_dict[10][1] - points_dict[175][1])
  df['face_w_dist'] = abs(points_dict[10][0] - points_dict[175][0])
  return df

def create_pose_features_dict(image, results, df):  
  width, height, color = image.shape
  mp_pose = mp.solutions.pose
  df['r_shoulder_lip'] = euc_distance([results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y * height],\
      [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height])
  df['r_shoulder_cheek'] = euc_distance([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * height],\
      [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height])
  df['r_shoulder_eye'] = euc_distance([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * height],\
      [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height])
  df['r_shoulder_eye_h'] = -results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * height + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
  df['r_shoulder_eye_w'] = - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * width + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width
  df['l_shoulder_lip'] = euc_distance([results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y * height],\
      [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height])
  df['l_shoulder_cheek'] = euc_distance([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * height],\
      [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height])
  df['l_shoulder_eye'] = euc_distance([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * height],\
      [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height])
  df['l_shoulder_eye_h'] = - results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * height + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
  df['l_shoulder_eye_w'] = - results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * width + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width  
  return df

def stat_vis(df):
    fig, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8, 1, figsize=(10, 8), sharex=True)
    x = np.array(list(df.index))
    try:
        y1 = df['emotion_neutral'].values
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
        df['emotion_neutral'] = 0
        y1 = df['emotion_neutral'].values
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
        y2 = df['emotion_happy'].values
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
        df['emotion_happy'] = 0
        y2 = df['emotion_happy'].values
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
        y3 = df['emotion_sad'].values
        ax3 = sns.barplot(x=x, y=y3, palette="rocket", ax=ax3)
        for bar in ax3.patches:
            ax3.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax3.axhline(0, color="k", clip_on=False)
        ax3.set_ylabel("sad")
    except:
        df['emotion_sad'] = 0
        y3 = df['emotion_sad'].values
        ax3 = sns.barplot(x=x, y=y3, palette="rocket", ax=ax3)
        for bar in ax3.patches:
            ax3.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax3.axhline(0, color="k", clip_on=False)
        ax3.set_ylabel("sad")

    try:
        y4 = df['emotion_surprise'].values
        ax4 = sns.barplot(x=x, y=y4, palette="rocket", ax=ax4)
        for bar in ax4.patches:
            ax4.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax4.axhline(0, color="k", clip_on=False)
        ax4.set_ylabel("surprise")
    except:
        df['emotion_surprise'] = 0
        y4 = df['emotion_surprise'].values
        ax4 = sns.barplot(x=x, y=y4, palette="rocket", ax=ax4)
        for bar in ax4.patches:
            ax4.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax4.axhline(0, color="k", clip_on=False)
        ax4.set_ylabel("surprise")

    try:
        y5 = df['emotion_disgust'].values
        ax5 = sns.barplot(x=x, y=y5, palette="rocket", ax=ax5)
        for bar in ax5.patches:
            ax5.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax5.axhline(0, color="k", clip_on=False)
        ax5.set_ylabel("disgust")
    except:
        df['emotion_disgust'] = 0
        y5 = df['emotion_disgust'].values
        ax5 = sns.barplot(x=x, y=y5, palette="rocket", ax=ax5)
        for bar in ax5.patches:
            ax5.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax5.axhline(0, color="k", clip_on=False)
        ax5.set_ylabel("disgust")

    try:
        y6 = df['emotion_NO FACE'].values
        ax6 = sns.barplot(x=x, y=y6, palette="rocket", ax=ax6)
        for bar in ax6.patches:
            ax6.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax6.axhline(0, color="k", clip_on=False)
        ax6.set_ylabel("NO FACE")
    except:
        df['emotion_NO FACE'] = 0
        y6 = df['emotion_NO FACE'].values
        ax6 = sns.barplot(x=x, y=y6, palette="rocket", ax=ax6)
        for bar in ax6.patches:
            ax6.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax6.axhline(0, color="k", clip_on=False)
        ax6.set_ylabel("NO FACE")
    try:
        y7 = df['sleep_predict'].values
        ax7 = sns.barplot(x=x, y=y7, palette="rocket", ax=ax7)
        for bar in ax7.patches:
            ax7.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax7.axhline(0, color="k", clip_on=False)
        ax7.set_ylabel("Sleep")
    except:
        df['sleep_predict'] = 0
        y7 = df['sleep_predict'].values
        ax7 = sns.barplot(x=x, y=y7, palette="rocket", ax=ax7)
        for bar in ax7.patches:
            ax7.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax7.axhline(0, color="k", clip_on=False)
        ax7.set_ylabel("Sleep")

    try:
        y8 = df['cheater'].values
        ax8 = sns.barplot(x=x, y=y8, palette="rocket", ax=ax8)
        for bar in ax8.patches:
            ax8.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax8.axhline(0, color="k", clip_on=False)
        ax8.set_ylabel("cheater")
    except:
        df['cheater'] = 0
        y8 = df['cheater'].values
        ax8 = sns.barplot(x=x, y=y8, palette="rocket", ax=ax8)
        for bar in ax8.patches:
            ax8.annotate(format(bar.get_height(), '.2f'),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 5),
            textcoords='offset points')
        ax8.axhline(0, color="k", clip_on=False)
        ax8.set_ylabel("cheater")

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(fig.axes, yticks=[])
    plt.tight_layout(h_pad=2)
    st.pyplot(fig)