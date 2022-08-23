import os

# main experiments
# os.system("python hybrid.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist")  # finish
# os.system("python hybrid.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist") # finish
os.system("python hybrid.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10")    # finish

# ablation study on segments
# finish
# os.system("python ablation_segment.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist")
# os.system("python ablation_segment.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist")
# os.system("python ablation_segment.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10")

# evaluation on ablation study on segments
# finish
# for exp in range(20):
#     os.system("python test_ablation_seg.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -g 1 --exp exp_{}".format(str(exp)))
#     os.system("python test_ablation_seg.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -g 1 --exp exp_{}".format(str(exp)))
#     os.system("python test_ablation_seg.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -g 1 --exp exp_{}".format(str(exp))) 


# ablation study on smoothness
# finish
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist --wt 0")
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist --wt 1")
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist --wt 0")
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist --wt 1")
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 --wt 0")
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 --wt 1")


# evaluation ablation study on smoothness
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -g 1 --exp without_smoothness")
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -g 1 --exp without_tl")
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -g 1 --exp without_smoothness")
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -g 1 --exp without_tl")
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -g 1 --exp without_smoothness")
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -g 1 --exp without_tl")

# test hybird
# os.system("python test.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist")
# os.system("python test.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist")
os.system("python test.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10")