
#! Noted that
#! iteration in frontend is start+(iteration -1)*period

# TODO change to timevis format
# TODO set a base class for some trainer functions... we dont need too many hyperparameters for frontend
# from unicodedata import name
from select import EPOLL_CLOEXEC
from flask import request, Response, Flask, jsonify, make_response
from flask_cors import CORS, cross_origin

import os
import sys
import json
import torch
import numpy as np
import tensorflow as tf
from umap.umap_ import find_ab_params

from antlr4 import *

sys.path.append("..")
from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.data import DataProvider
from singleVis.eval.evaluator import Evaluator
from singleVis.trainer import SingleVisTrainer
from singleVis.losses import ReconstructionLoss, UmapLoss, SingleVisLoss
from singleVis.visualizer import visualizer
from BackendAdapter import TimeVisBackend
from utils import *

# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/updateProjection', methods=["POST", "GET"])
@cross_origin()
def update_projection():
    res = request.get_json()
    CONTENT_PATH = os.path.normpath(res['path'])
    EPOCH = int(res['iteration'])
    predicates = res["predicates"]

    sys.path.append(CONTENT_PATH)
    timevis = initialize_backend(CONTENT_PATH)

    embedding_2d, grid, decision_view, label_color_list, label_list, max_iter, current_index, \
    testing_data_index, eval_new, prediction_list, selected_points = update_epoch_projection(timevis, EPOCH, predicates)

    sys.path.remove(CONTENT_PATH)

    return make_response(jsonify({'result': embedding_2d, 'grid_index': grid, 'grid_color': decision_view,
                                  'label_color_list': label_color_list, 'label_list': label_list,
                                  'maximum_iteration': max_iter, 'training_data': current_index,
                                  'testing_data': testing_data_index, 'evaluation': eval_new,
                                  'prediction_list': prediction_list,
                                  "selectedPoints":selected_points.tolist()}), 200)

@app.route('/query', methods=["POST"])
@cross_origin()
def filter():
    res = request.get_json()
    CONTENT_PATH = os.path.normpath(res['path'])
    EPOCH = int(res['iteration'])
    predicates = res["predicates"]

    sys.path.append(CONTENT_PATH)
    timevis = initialize_backend(CONTENT_PATH)

    training_data_number = timevis.hyperparameters["TRAINING"]["train_num"]
    testing_data_number = timevis.hyperparameters["TRAINING"]["test_num"]

    current_index = timevis.get_epoch_index(EPOCH)
    selected_points = np.arange(training_data_number + testing_data_number)[current_index]
    for key in predicates.keys():
        if key == "label":
            tmp = np.array(timevis.filter_label(predicates[key]))
        elif key == "type":
            tmp = np.array(timevis.filter_type(predicates[key], int(EPOCH)))
        else:
            tmp = np.arange(training_data_number + testing_data_number)
        selected_points = np.intersect1d(selected_points, tmp)
    sys.path.remove(CONTENT_PATH)

    return make_response(jsonify({"selectedPoints": selected_points}), 200)


# @app.route('/save_human_selections', methods=["POST"])
# @cross_origin()
# def save_human_selections():
#     data = request.get_json()
#     indices = data["newIndices"]
#     CONTENT_PATH = os.path.normpath(data['content_path'])
#     iteration = data["iteration"]
#     sys.path.append(CONTENT_PATH)

#     timevis = initialize_backend(CONTENT_PATH)
#     timevis.save_human_selection(iteration, indices)
#     sys.path.remove(CONTENT_PATH)
#     return make_response(jsonify({"message":"Save user selection succefully!"}), 200)

@app.route('/sprite', methods=["POST","GET"])
@cross_origin()
def sprite_image():
    path= request.args.get("path")
    sprite = tf.io.gfile.GFile(path, "rb")
    encoded_image_string = sprite.read()
    sprite.close()
    image_type = "image/png"
    return Response(encoded_image_string, status=200, mimetype=image_type)

@app.route('/al_query', methods=["POST"])
@cross_origin()
def al_query():
    data = request.get_json()
    CONTENT_PATH = os.path.normpath(data['content_path'])
    iteration = data["iteration"]
    strategy = data["strategy"]
    sys.path.append(CONTENT_PATH)

    timevis = initialize_backend(CONTENT_PATH)
    indices = timevis.al_query(iteration, strategy)

    sys.path.remove(CONTENT_PATH)
    return make_response(jsonify({"selectedPoints": indices}), 200)

@app.route('/al_train', methods=["POST"])
@cross_origin()
def al_train():
    data = request.get_json()
    CONTENT_PATH = os.path.normpath(data['content_path'])
    new_indices = data["newIndices"]
    iteration = data["iteration"]
    strategy = data["strategy"]
    sys.path.append(CONTENT_PATH)

    timevis = initialize_backend(CONTENT_PATH)
    timevis.al_train(iteration, strategy, new_indices)

    # preprocess
    NEW_ITERATION = iteration + 1
    PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
    B_N_EPOCHS = config["VISUALIZATION"]["BOUNDARY"]["B_N_EPOCHS"]
    L_BOUND = config["VISUALIZATION"]["BOUNDARY"]["L_BOUND"]
    if PREPROCESS:
        timevis.data_provider._meta_data(iteration+1)
        if B_N_EPOCHS != 0:
            LEN = len(timevis.data_provider.train_labels(NEW_ITERATION))
            timevis.data_provider._estimate_boundary(NEW_ITERATION, LEN//10, l_bound=L_BOUND)

    # train visualization model
    # TODO
    
    # update iteration projection
    embedding_2d, grid, decision_view, label_color_list, label_list, max_iter, current_index, \
    testing_data_index, eval_new, prediction_list, selected_points = update_epoch_projection(timevis, NEW_ITERATION, dict())

    sys.path.remove(CONTENT_PATH)
    return make_response(jsonify({'result': embedding_2d, 'grid_index': grid, 'grid_color': decision_view,
                                  'label_color_list': label_color_list, 'label_list': label_list,
                                  'maximum_iteration': max_iter, 'training_data': current_index,
                                  'testing_data': testing_data_index, 'evaluation': eval_new,
                                  'prediction_list': prediction_list,
                                  "selectedPoints":selected_points.tolist()}), 200)

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
        ip_adress = config["ServerIP"]
        port = config["ServerPort"]
    app.run(host=ip_adress, port=int(port))
