import os
################################################################################################
#                                                                                              #
#                                         Data Prepare                                         #
#                                                                                              #
################################################################################################
# Create visualization with two strategies

# timevis on noisy dataset
datasets = ["cifar10","mnist","fmnist"]
rates = ["5", "10", "20"]
for data in datasets:
    for rate in rates:
        os.system("python timevis_main.py --content_path /home/xianglin/projects/DVI_data/noisy/symmetric/{}/{}".format(data, rate))
# timevis on active learning dataset
datasets = ["CIFAR10", "FMNIST", "MNIST"]
rates = ["10", "20", "30"]
for data in datasets:
    for rate in rates:
        os.system("python timevis_main.py --content_path /home/xianglin/projects/DVI_data/active_learning/random/resnet18/{}/{}".format(data, rate))
################################################################################################



################################################################################################
#                                                                                              #
#                                  RQ1: Anomaly Detection                                      #
#                                                                                              #
################################################################################################
os.system("python anomaly_detection.py --dataset mnist --method tfDVI")
os.system("python anomaly_detection.py --dataset mnist --method TimeVis")
os.system("python anomaly_detection.py --dataset fmnist --method tfDVI")
os.system("python anomaly_detection.py --dataset fmnist --method TimeVis")
os.system("python anomaly_detection.py --dataset cifar10 --method tfDVI")
os.system("python anomaly_detection.py --dataset cifar10 --method TimeVis")
################################################################################################



################################################################################################
#                                                                                              #
#                                  RQ2: Feedback Simulation                                    #
#                                                                                              #
################################################################################################
# feedback on noisy dataset
datasets = ["mnist", "fmnist", "cifar10"]
rates = ["5", "10", "20"]
methods = ["tfDVI", "TimeVis"]
for data in datasets:
    for rate in rates:
        for method in methods:
            os.system("python feedback_noise.py --dataset {} --noise_rate {} --method {}".format(data, rate, method))

# feedback on active learning dataset
rates = ["10", "20", "30"]
for data in datasets:
    for rate in rates:
        for method in methods:
            os.system("python feedback_al.py --dataset {} --rate {} --method {}".format(data.upper(), rate, method))
################################################################################################



################################################################################################
#                                                                                              #
#                                    RQ3: Error Resistence                                     #
#                                                                                              #
################################################################################################
repeat = 4
for _ in range(repeat):
    # # feedback test on noisy dataset
    datasets = ["mnist", "fmnist", "cifar10"]
    rates = ["5", "10", "20"]
    for data in datasets:
        for rate in rates:
            os.system("python feedback_noise_test.py --dataset {} --noise_rate {} --tolerance 0.03 0.05 0.1 0.15 --repeat 2 --round 50".format(data, rate))

    # feedback test on active learning dataset
    rates = ["10", "20", "30"]
    for data in datasets:
        for rate in rates:
            os.system("python feedback_al_test.py --dataset {} --rate {} --tolerance 0.03 0.05 0.1 0.15 --repeat 1 --round 50".format(data, rate))
