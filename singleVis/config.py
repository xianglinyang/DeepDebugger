"""
Training config for different datasets.
"""

dataset_config = {
    "cifar10": {
        "NET": "resnet18",
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.25,#0.2
        "ALPHA":0,
        "BETA":1,#.1
        "HIDDEN_LAYER":3,
        "INIT_NUM":100,
        "EPOCH_START": 1,
        "EPOCH_END": 11,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15, 
            "MAX_EPOCH": 30,
            "S_N_EPOCHS": 3,
            "B_N_EPOCHS":3, # 1
            "T_N_EPOCHS": 100,
            "PATIENT": 3,    # early stopping patient
        }
        
    },
    "cifar10_n": {
        "NET": "resnet18",
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.4, # 0.2
        "ALPHA":0,#0
        "BETA":.1,#.1
        "HIDDEN_LAYER":4,
        "INIT_NUM":300,
        "EPOCH_START": 40,
        "EPOCH_END": 200,
        "EPOCH_PERIOD": 4,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":5,
            "T_N_EPOCHS": 100,
            "PATIENT": 3,    # early stopping patient
        }
        
    },
    "mnist": {
        "NET": "resnet18",
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "L_BOUND":0.4,
        "MAX_HAUSDORFF":0.28,
        "ALPHA":0,
        "BETA":1,
        "HIDDEN_LAYER":3,
        "INIT_NUM":200,
        "EPOCH_START": 10,
        "EPOCH_END": 10,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 30,
            "S_N_EPOCHS": 3,
            "B_N_EPOCHS":3,#1
            "T_N_EPOCHS": 100,
            "PATIENT": 3,    # early stopping patient
        }
    },
    "fmnist": {
        "NET":"resnet18",
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":20.,
        "L_BOUND":0.5,
        "MAX_HAUSDORFF":.07,#.06
        "ALPHA":0,#2
        "BETA":1.3,
        "INIT_NUM":200,
        "HIDDEN_LAYER":3,
        "EPOCH_START": 10,
        "EPOCH_END": 10,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 10,
            "S_N_EPOCHS": 3,
            "B_N_EPOCHS":5,
            "T_N_EPOCHS": 50,
            "PATIENT": 4,    # early stopping patient
        }
    },
    "mnist_mlp": {
        "NET":"mlp3",
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "L_BOUND":0.5,
        "MAX_HAUSDORFF":0.25,
        "ALPHA":1,
        "BETA":1,
        "INIT_NUM":300,
        "HIDDEN_LAYER":3,
        "EPOCH_START": 1,
        "EPOCH_END": 15,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 6,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":5,#1
            "T_N_EPOCHS": 100,
            "PATIENT": 3,    # early stopping patient
        }
    },
    # hyperparameter for sequence selection
    "mnist_full": {
        "NET":"resnet18",
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "L_BOUND":0.4,
        "MAX_HAUSDORFF":0.25,
        "ALPHA":1,# 1
        "BETA":1,#1
        "HIDDEN_LAYER":3,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 20,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":5,
            "T_N_EPOCHS": 20,
            "PATIENT": 3,    # early stopping patient
        }
    },
    "fmnist_full": {
        "NET":"resnet18",
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":30.,
        "L_BOUND":0.5,
        "MAX_HAUSDORFF":.07,
        "ALPHA":2,#2
        "BETA":1.3,#1.3
        "HIDDEN_LAYER":4,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 50,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 3,
            "B_N_EPOCHS": 4,
            "T_N_EPOCHS": 80,
            "PATIENT": 4,    # early stopping patient
        }
    },
    "cifar10_full": {
        "NET": "resnet18",
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.2, # 0.2
        "ALPHA":0,#0
        "BETA":.1,#.1
        "HIDDEN_LAYER":4,
        "INIT_NUM":300,
        "EPOCH_START": 40,
        "EPOCH_END": 200,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":5,
            "T_N_EPOCHS": 100,
            "PATIENT": 3,    # early stopping patient
        }
        
    },
    # hyperparameter for parallel selection
    "mnist_p":{
        "NET":"resnet18",
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "L_BOUND":0.4,
        "MAX_HAUSDORFF":0.28,
        "ALPHA":0,# 1
        "BETA":1,#1
        "HIDDEN_LAYER":4,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 20,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS": 5,
            "T_N_EPOCHS": 10,
            "PATIENT": 3,    # early stopping patient
        }
    },
}