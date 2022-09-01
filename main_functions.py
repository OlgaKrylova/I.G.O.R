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

    