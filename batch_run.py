import os

# main experiments
# os.system("python hybrid.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist")
# os.system("python hybrid.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist")
# os.system("python hybrid.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10")

# ablation study on segments
# os.system("python ablation_segment.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist")#finish
# os.system("python ablation_segment.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist")#finish
# os.system("python ablation_segment.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10") # on going 12

# evaluation on ablation study on segments
# for exp in range(20):
    # os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -g 0 --exp exp_{}".format(str(exp)))#finish
    # os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -g 0 --exp exp_{}".format(str(exp)))#finish
    # os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -g 0 --exp exp_{}".format(str(exp))) # not yet


# ablation study on smoothness
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist --wt 0")# ongoing 0
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist --wt 1")# ongoing 1
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist --wt 0")# ongoing 2
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist --wt 1")# ongoing 3
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 --wt 0")# ongoing 4
# os.system("python ablation_smoothness.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 --wt 1")# ongoing 5


# evaluation ablation study on smoothness
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -g 0 --exp without_smoothness")# not yet
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -g 0 --exp without_tl")# not yet
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -g 0 --exp without_smoothness")# not yet
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -g 0 --exp without_tl")# not yet
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -g 0 --exp without_smoothness")# not yet
# os.system("python test_ablation.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -g 0 --exp without_tl")# not yet