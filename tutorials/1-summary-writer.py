'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# >>>>>>>>>> Define summary writer
from writer.summary_writer import SummaryWriter
log_dir = "path/to/content" # User define
writer = SummaryWriter(log_dir)
# <<<<<<<<<< Define summary writer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
record_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# >>>>>>>>>>Record Data
writer.add_training_data(record_trainset) # use test_transform
writer.add_testing_data(testset)
# <<<<<<<<<<Record Data

print('==> Building model..')
net = ResNet18()    # choose your own model
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)

# Training
def train():
    net.train()
    for _, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


prev_id = None
idxs = list(range(len(trainset)))
for epoch in range(1,200,1):
    train()
    if epoch % 10 == 0:
        # >>>>>>>>>>record checkpoint for every 10 epochs
        writer.add_checkpoint_data(net.state_dict(), idxs, prev_id)
        # <<<<<<<<<<record checkpoint for every 10 epochs
    prev_id = epoch

# >>>>>>>>>> Record Config
config_dict = {
    "SETTING": "normal",
    "CLASSES": classes, 
    "GPU":"1",
    "DATASET": "cifar10",
    "EPOCH_START": 1,
    "EPOCH_END": 200,
    "EPOCH_PERIOD": 1,
    "TRAINING": {
        "NET": "resnet18",
        "num_class": 10,
        "train_num": 60000,
        "test_num": 10000,
    },
    "VISUALIZATION":{
        "PREPROCESS":1,
        "BOUNDARY":{
            "B_N_EPOCHS": 0,
            "L_BOUND":0.5,
        },
        "INIT_NUM": 300,
        "ALPHA":1,
        "BETA":1,
        "MAX_HAUSDORFF":0.33,
        "LAMBDA": 1,
        "S_LAMBDA": 1,
        "ENCODER_DIMS":[512,256,256,256,2],
        "DECODER_DIMS":[2,256,256,256,512],
        "N_NEIGHBORS":15,
        "MAX_EPOCH": 20,
        "S_N_EPOCHS": 5,
        "T_N_EPOCHS": 20,
        "PATIENT": 3,
        "RESOLUTION":300,
        "VIS_MODEL_NAME": "DeepDebugger",
        "EVALUATION_NAME": "test_evaluation_DeepDebugger"
    }
}
# <<<<<<<<<< Record Config

# >>>>>>>>>> Choose a visualization method to visualize embedding
from singleVis.Strategy import DeepDebugger
dd = DeepDebugger(config)
dd.visualize_embedding()
# <<<<<<<<<< Choose a visualization method to visualize embedding


# Next start server and frontend




