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
def create_features_dict(image, points, points_dict, P, s = 0):
  df = {}
  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  if results.multi_face_landmarks:
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
      df['teta1', 'teta2', 'teta3', 'teta4', 'teta5', 'teta6', 'teta7', 'teta8', 'teta9', 'teta10', 'l_eye_w',\
              'l_eye_h', 'r_eye_w', 'r_eye_h', 'lips_w', 'lips_h', 'lips_h_in', 'brows_dist', 'r_cheek_eye', 'l_cheek_eye',\
              'r_cheek_lip', 'l_cheek_lip', 'r_eye_brow_in', 'l_eye_brow_in', 'r_eye_brow_out', 'l_eye_brow_out', 'r_eye_nose_in',\
                'l_eye_nose_in'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
faces_data_frame = pd.DataFrame(df,index=[s])
return faces_data_frame