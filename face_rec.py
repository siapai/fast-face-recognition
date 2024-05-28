import numpy as np
import pandas as pd
import cv2

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to redis
hostname = 'localhost'
port = 6379

r = redis.StrictRedis(host=hostname, port=port)

# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


def ml_search_algorithm(data_frame, feature_column, test_vector, name_role=None, thresh=0.5):
    """
    Cosine similarity base search algorithm
    """

    if name_role is None:
        name_role = ['Name', 'Role']

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
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


def face_prediction(test_image, data_frame, feature_column, name_role):
    # Step1: take the test image and apply to insight face
    result = faceapp.get(test_image)
    test_copy = test_image.copy()

    # Step2: use for loop to extract each embedding and pass to ml_search_algorithm
    for res in result:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(
            data_frame,
            feature_column,
            test_vector=embeddings,
            name_role=name_role
        )
        if person_name != 'Unknown':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
        text_gen = person_name
        cv2.putText(test_copy, text_gen, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

    return test_copy

