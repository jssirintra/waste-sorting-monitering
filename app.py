from flask import Flask, request, jsonify 
import cv2 
import numpy as np
from ultralytics import YOLO 
from flask_apscheduler import APScheduler
from google.cloud import storage
from google.oauth2 import service_account
from datetime import date
import pandas as pd

# app = Flask(__name__)
# scheduler = APScheduler()

model = YOLO('best.pt')

gcp_credentials_file = './monitor-waste-video-acc.json'
gcp_project = 'zero-waste-deploy2'
gcp_bucket = 'monitor-waste-video'
credentials = service_account.Credentials.from_service_account_file(gcp_credentials_file)
storage_client = storage.Client(project=gcp_project, credentials=credentials)
bucket = storage_client.get_bucket('monitor-waste-result')
bucket_algo = storage_client.get_bucket('monitor-waste-result-modified')

# Detect objects in an image
def detect_objects(image_path,day):
    ###for video
    cap = cv2.VideoCapture(image_path)
    # get video dimensions
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # day = date.today()
    video = image_path.split('/')[5].split('.')[0]
    frame_num = 1
    frame_list = []
    track_id_list = []
    class_name = []
    class_id = []
    conf_list = []
    bx = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break                                                      

        # predictions = model(frame, save=False,save_txt=False,stream=True)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf = 0.25)

        # for result in predictions:
        #         boxes = result.boxes.cpu().numpy()
        #         for box in boxes:
        #             frame_list.append(frame_num)
        #             clss = int(box.cls[0])
        #             class_id.append(clss)
        #             class_name.append(model.names[clss])
        #             conf.append(int(box.conf[0]*100))
        #             bx.append(box.xyxy.tolist())
        # frame_num = frame_num+1

        for result in results:
            boxes = result.boxes.xywh.cpu()
            num = len(boxes)
            frame_list.extend([frame_num]*num)

            try:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                track_id_list.extend(track_ids)
            except AttributeError:
                track_id_list.extend([None]*num)

            try:
                class_ids = np.array(result.boxes.cls).astype(int)
                class_id.extend(class_ids)
            except AttributeError:
                class_id.extend([None]*num)

            try:
                conf = np.array(results[0].boxes.conf)
                conf_list.extend(conf)
            except AttributeError:
                conf_list.extend([None]*num)
            
            try:
                xy = np.array(boxes)
                bx.extend(xy)
            except NameError:
                bx.extend([None]*num)
            
        frame_num += 1


    detections_json = {
        "date": day,
        "time": video,
        "frame_list": frame_list,
        "track_id_list": track_id_list,
        "class_id": class_id,
        "confidence": conf_list,
        "box_coord": bx
    }
    return detections_json
    
y1 = 580
x1 = 660
x2 = 1230

sub_class = {0: 'can',
            1: 'dirty food container',
            2: 'filled plastic bag',
            3: 'food container',
            4: 'milk carton',
            5: 'paper bag',
            6: 'plastic bag',
            7: 'plastic bottle',
            8: 'plastic cup',
            9: 'plastic cutlery',
            10: 'snack packaging',
            11: 'sticks',
            12: 'straw',
            13: 'tissue paper',
            14: 'zero waste cup'}

def map_trash(x):
    if x< x1:
        return "garbage"
    elif (x>= x1)&(x<x2):
        return "recycle_plus"
    elif x>=x2:
        return "pet"
    else: return None

def map_in_out(y):
    if y> y1:
        return "in"
    else:
        return "out"

def get_waste_type(class_id):
  match class_id:
    case 0 | 7:
      return "pet"
    case 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12 :
      return "recycle_plus"
    case 13:
      return "garbage"
    case 14:
      return "zero_waste"
    case _:
      return "always_wrong"

# def verify(df):
#   #df = pd.read_csv("track_result.csv")
#     groupby1 = df.groupby('track_id_list').agg(
#         frame_min=('frame_list', 'min'),
#         frame_max=('frame_list', 'max'),
#         unique_classes=('class_id', 'nunique'),
#         classes=('class_id', 'min'),
#         frame_count=('frame_list', 'count')).reset_index()
#     groupby1['frame_duration'] = groupby1['frame_max']-groupby1['frame_min']

#     duration = 10
#     new_object = groupby1[(groupby1['frame_min']>1)&(groupby1['frame_duration']>duration)]
#     new_obj_with_coor = new_object[['track_id_list','frame_max']].merge(df, how = 'left',left_on=['track_id_list','frame_max'], right_on=['track_id_list','frame_list'])
#     new_obj_with_coor['Current type'] = new_obj_with_coor['box_coord'].apply(lambda x: map_trash(x[0]))
#     new_obj_with_coor['Correct type'] = new_obj_with_coor['class_id'].apply(lambda x: get_waste_type(x))
#     new_obj_with_coor['Sub category'] = new_obj_with_coor['class_id'].apply(lambda x: sub_class[x])
#     new_obj_with_coor['Correctness'] = new_obj_with_coor['Current type']==new_obj_with_coor['Correct type']
#     new_obj_with_coor.columns = ['object_id', 'frame_max', 'date', 'time', 'frame_list', 'class_id',
#         'conf_list', 'box_coord', 'Current type', 'Correct type', 'Sub category',
#         'Correctness']
#     return new_obj_with_coor

def get_most_common_class(df):
    # df['box_coord'] = df['box_coord'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df['x_coor'] = df['box_coord'].apply(lambda x: map_trash(x[0]))
    df['y_coor'] = df['box_coord'].apply(lambda x: map_in_out(x[1]))

    df['frame_min'] = df.groupby('track_id_list')['frame_list'].transform(np.min)
    df['frame_max'] = df.groupby('track_id_list')['frame_list'].transform(np.max)
    df = df[df['frame_min']!=1].reset_index(drop=True)
    df['frame_duration'] = df['frame_max']-df['frame_min']+1
    df['class_id'] = df.groupby('track_id_list')['class_id'].transform(lambda x: x.mode()[0])

    sorted_df = df.sort_values(by=['track_id_list', 'frame_list']).reset_index(drop=True)

    #keep all instances for each id where it is in a certain class for more than 10 frames
    sorted_df['group_x_coor'] = sorted_df.groupby('track_id_list')['x_coor'].transform(lambda x: (x != x.shift()).cumsum())
    group_sizes = sorted_df.groupby(['track_id_list', 'x_coor', 'group_x_coor']).size()
    large_groups = group_sizes[group_sizes >= 10].reset_index()[['track_id_list', 'x_coor', 'group_x_coor']]
    filtered_df = pd.merge(sorted_df, large_groups, on=['track_id_list', 'x_coor', 'group_x_coor'])

    final_df = filtered_df.groupby('track_id_list').apply(lambda x: x.loc[x['group_x_coor'].idxmax()]).reset_index(drop=True)

    final_df['Correct type'] = final_df['class_id'].apply(lambda x: get_waste_type(x))
    final_df['Sub category'] = final_df['class_id'].apply(lambda x: sub_class[x])
    final_df['Correctness'] = final_df['x_coor']==final_df['Correct type']

    final_df.drop(['y_coor','group_x_coor'], axis=1, inplace=True)

    final_df.columns = ['date', 'time','frame_list','object_id', 'class_id','conf_list','box_coord','Current type',
                        'frame_min','frame_max','frame_duration','Correct type', 'Sub category','Correctness']
    final_df = final_df[['date', 'time','object_id', 'class_id','conf_list','box_coord', 'Sub category','Current type',
                        'Correct type','Correctness','frame_min','frame_max','frame_duration','frame_list']]

    return final_df




#@scheduler.task('cron', id='my_job', day_of_week="mon,tue,wed,thu,fri", hour=14, minute=12, second=0, timezone="Asia/Bangkok")
def my_job():

    today = date.today()
    today = str(today)
    video_list = []

    for blob in storage_client.list_blobs('monitor-waste-video', prefix=today):
        name = str(blob.name)
        if((name+".").split(".")[1]=='mp4'):
            video_list.append(name)
    # video_list = ['2024-06-25/10-01-47.mp4']

    print('-----------------------\n---starting---: date ' + str(today))
    detections_df = pd.DataFrame()
    for video in video_list:
        df = pd.DataFrame(detect_objects('https://storage.googleapis.com/monitor-waste-video/' + video,today))
        print('---detection done for',str(video) , 'video---')
        detections_df = pd.concat([detections_df, df], ignore_index=True)
    print('--start uploading to cloud---')

    #detections_df.to_csv('detect.csv')
    bucket.blob(str(today)+".csv").upload_from_string(detections_df.to_csv(), 'text/csv')
    print('--uploaded to cloud---: date '+ str(today))

    #get_most_common_class(detections_df).to_csv('detect_algo.csv')
    bucket_algo.blob(str(today)+".csv").upload_from_string(get_most_common_class(detections_df).to_csv(), 'text/csv')
    print('--uploaded modified to cloud---: date '+ str(today))





if __name__ == '__main__':
    # Load YOLO model
    # yolo_net, classes = load_yolo()
    # scheduler.init_app(app)
    # Start Flask application
    # scheduler.start()
    my_job()
    # app.run(host="0.0.0.0", port=5000,debug=True)
    
