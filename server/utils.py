import os
import json
import time
import csv
import numpy as np
import sys
import pickle
import base64

vis_path = ".."
sys.path.append(vis_path)
from context import VisContext, ActiveLearningContext, AnormalyContext
from strategy import DeepDebugger, TimeVis, DeepVisualInsight

"""Interface align"""

def initialize_strategy(CONTENT_PATH, VIS_METHOD):
    # initailize strategy (visualization method)
    with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
        conf = json.load(f)
    config = conf[VIS_METHOD]

    if VIS_METHOD == "DVI":
        strategy = DeepVisualInsight(CONTENT_PATH, config)
    elif VIS_METHOD == "TimeVis":
        strategy = TimeVis(CONTENT_PATH, config)
    elif VIS_METHOD == DeepDebugger(CONTENT_PATH, config):
        strategy = DeepDebugger(CONTENT_PATH, config)
    else:
        raise NotImplementedError
    return strategy

def initialize_context(strategy, setting):
    if setting == "normal":
        context = VisContext(strategy)
    elif setting == "active learning":
        context = ActiveLearningContext(strategy)
    elif setting == "dense al":
        context == ActiveLearningContext(strategy)
    elif setting == "abnormal":
        context = AnormalyContext(strategy, 80)
    else:
        raise NotImplementedError

def initialize_backend(CONTENT_PATH, VIS_METHOD, SETTING):
    """ initialize backend for visualization

    Args:
        CONTENT_PATH (str): the directory to training process
        VIS_METHOD (str): visualization strategy
            "DVI", "TimeVis", "DeepDebugger",...
        setting (str): context
            "normal", "active learning", "dense al", "abnormal"

    Raises:
        NotImplementedError: _description_

    Returns:
        backend: a context with a specific strategy
    """
    strategy = initialize_strategy(CONTENT_PATH, VIS_METHOD)
    context = initialize_context(strategy=strategy, setting=SETTING)
    return context


def update_epoch_projection(context, EPOCH, predicates):

    train_data = context.strategy.data_provider.train_representation(EPOCH)
    test_data = context.strategy.data_provider.test_representation(EPOCH)
    all_data = np.concatenate((train_data, test_data), axis=0)

    embedding_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "embedding.npy")
    
    if os.path.exists(embedding_path):
        embedding_2d = np.load(embedding_path)
    else:
        embedding_2d = context.strategy.projector.batch_project(EPOCH, all_data)
        np.save(embedding_path, embedding_2d)

    train_labels = context.strategy.data_provider.train_labels(EPOCH)
    test_labels = context.strategy.data_provider.test_labels(EPOCH)
    labels = np.concatenate((train_labels, test_labels), axis=0).tolist()

    training_data_number = context.strategy.config["TRAINING"]["train_num"]
    testing_data_number = context.strategy.config["TRAINING"]["test_num"]
    training_data_index = list(range(training_data_number))
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))

    # return the image of background
    # read cache if exists
    bgimg_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "bgimg.png")
    grid_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "grid.pkl")
    if os.path.exists(bgimg_path) and os.path.exists(grid_path):
        with open(os.path.join(grid_path), "rb") as f:
            grid = pickle.load(f)
        with open(bgimg_path, 'rb') as img_f:
            img_stream = img_f.read()
        b_fig = base64.b64encode(img_stream).decode()
    else:
        x_min, y_min, x_max, y_max, b_fig = context.strategy.vis.get_background(EPOCH, context.strategy.config["VISUALIZATION"]["RESOLUTION"])
        grid = [x_min, y_min, x_max, y_max]
        # formating
        grid = [float(i) for i in grid]
        b_fig = str(b_fig, encoding='utf-8')

    # save results, grid and decision_view
    save_path = context.strategy.data_provider.checkpoint_path(EPOCH)
    with open(os.path.join(save_path, "grid.pkl"), "wb") as f:
        pickle.dump(grid, f)
    np.save(os.path.join(save_path, "embedding.npy"), embedding_2d)
    
    color = context.strategy.vis.get_standard_classes_color() * 255
    color = color.astype(int).tolist()

    # TODO fix its structure
    file_name = context.strategy.config["VISUALIZATION"]["EVALUATION_NAME"]
    evaluation = context.strategy.evaluator.get_eval(file_name=file_name)
    eval_new = dict()
    eval_new["train_acc"] = evaluation["train_acc"][str(EPOCH)]
    eval_new["test_acc"] = evaluation["test_acc"][str(EPOCH)]

    label_color_list = []
    label_list = []
    label_name_dict = dict()
    for i, label in enumerate(context.strategy.config["CLASSES"]):
        label_name_dict[i] = label
        
    for label in labels:
        label_color_list.append(color[int(label)])
        label_list.append(context.strategy.config["CLASSES"][int(label)])

    prediction_list = []
    prediction = context.strategy.data_provider.get_pred(EPOCH, all_data).argmax(1)

    for i in range(len(prediction)):
        prediction_list.append(context.strategy.config["CLASSES"][prediction[i]])
    
    max_iter = context.get_max_iter()
    
    # current_index = timevis.get_epoch_index(EPOCH)
    # selected_points = np.arange(training_data_number + testing_data_number)[current_index]
    selected_points = np.arange(training_data_number + testing_data_number)
    for key in predicates.keys():
        if key == "label":
            tmp = np.array(context.filter_label(predicates[key]))
        elif key == "type":
            tmp = np.array(context.filter_type(predicates[key], int(EPOCH)))
        else:
            tmp = np.arange(training_data_number + testing_data_number)
        selected_points = np.intersect1d(selected_points, tmp)
    
    properties = np.concatenate((np.zeros(training_data_number, dtype=np.int16), 2*np.ones(testing_data_number, dtype=np.int16)), axis=0)
    lb = context.get_epoch_index(EPOCH)
    ulb = np.setdiff1d(training_data_index, lb)
    properties[ulb] = 1
    
    return embedding_2d.tolist(), grid, b_fig, label_name_dict, label_color_list, label_list, max_iter, training_data_index, testing_data_index, eval_new, prediction_list, selected_points, properties


def add_line(path, data_row):
    """
    data_row: list, [API_name, username, time]
    """
    now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    data_row.append(now_time)
    with open(path, "a+") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)