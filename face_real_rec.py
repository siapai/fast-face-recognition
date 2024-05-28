import os
from dotenv import load_dotenv
import constants

import numpy as np
import pandas as pd
import cv2

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# time
import time
from datetime import datetime

load_dotenv()

# Connect to redis
hostname = os.environ["REDIS_HOST"]
port = 6379

r = redis.StrictRedis(host=hostname, port=port)


# Retrieve data from database
def retrieve_data(name):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_id', 'facial_features']
    retrieve_df[['Name', 'User_Id']] = retrieve_df['name_id'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name', 'User_Id', 'facial_features']]


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


def ml_search_algorithm(data_frame, feature_column, test_vector, name_id=None, thresh=0.5):
    """
    Cosine similarity base search algorithm
    """

    if name_id is None:
        name_id = ['Name', 'User_Id']

    # Step1: take the dataframe

    data_frame = data_frame.copy()

    # Step2: index face embedding and convert to array
    x_list = data_frame[feature_column].tolist()
    x = np.asarray(x_list)

    # Step3: calc cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    data_frame['cosine'] = similar_arr

    # Step4: filter the data
    data_filter = data_frame.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, user_id = data_filter.loc[argmax][name_id]
    else:
        person_name = 'Unknown'
        user_id = 'Unknown'

    return person_name, user_id


# Save logs every 1 min
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], user_id=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], user_id=[], current_time=[])

    def save_logs_redis(self):
        # Step1: create logs dataframe
        data_frame = pd.DataFrame(self.logs)

        # Step2: drop the duplicate information (distinct name)
        data_frame.drop_duplicates('name', inplace=True)

        # Step3: push data to redis database
        name_list = data_frame['name'].tolist()
        user_id_list = data_frame['user_id'].tolist()
        ctime_list = data_frame['current_time'].tolist()
        encoded_data = []
        for name, user_id, ctime in zip(name_list, user_id_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{user_id}@{ctime}"
                encoded_data.append(concat_string)
        if len(encoded_data) > 0:
            r.lpush(constants.LOG_KEY, *encoded_data)

        self.reset_dict()

    def face_prediction(self, test_image, data_frame, feature_column, name_id):
        # Step0: Calc time
        current_time = str(datetime.now())

        # Step1: take the test image and apply to insight face
        result = faceapp.get(test_image)
        test_copy = test_image.copy()

        # Step2: use for loop to extract each embedding and pass to ml_search_algorithm
        for res in result:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, user_id = ml_search_algorithm(
                data_frame,
                feature_column,
                test_vector=embeddings,
                name_id=name_id
            )
            if person_name != 'Unknown':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
            cv2.putText(test_copy, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
            # save info to logs dict
            self.logs['name'].append(person_name)
            self.logs['user_id'].append(user_id)
            self.logs['current_time'].append(current_time)

        return test_copy


# Registration Form
# get results from insightface model
class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        embeddings = None

        for res in results:
            self.sample += 1

            if self.sample <= 300:
                # facial features
                embeddings = res['embedding']

            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # sample text
            text = f'samples = {300 if self.sample > 300 else self.sample}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

        return frame, embeddings

    def save_data_to_redis(self, name, person_id):

        # validation
        if name is not None or person_id is not None:
            if name.strip() != '' or person_id.strip() != '':
                key = f'{name}@{person_id}'
            else:
                return False, 'invalid_name'
        else:
            return False, 'invalid_name'

        if f'{person_id}.txt' not in os.listdir():
            return False, 'embedding_not_exist'

        # load face_embedding.txt
        x_array = np.loadtxt(f'{person_id}.txt', dtype=np.float32)  # flatten array

        # convert int array
        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # calc mean
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # save into redis
        r.hset(name=constants.REGISTER_KEY, key=key, value=x_mean_bytes)

        # remove file
        os.remove(f'{person_id}.txt')
        self.reset()

        return True, 'OK'
