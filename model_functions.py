#import torch.nn.functional as F
import torch
from torch import nn, optim
from util_functions import load_json
from collections import OrderedDict
from torchvision import datasets,  transforms#, models
from torch.utils.data import DataLoader


def load_dataset(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(224),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = DataLoader(val_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)
    return train_data, trainloader, validloader, testloader

def build_classifier(arg):
    from torchvision import models          # need this here to prevent UnboundLocalError: local variable 'models' referenced before assignment
    resnet18 = models.resnet18(pretrained = True)
    alexnet = models.alexnet(pretrained = True)
    vgg16 = models.vgg16(pretrained = True)
    models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg16': vgg16}
                     
    # Build and train your network
    model = models[arg.arch]

    #model = models.(arg.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    input_unit = model.classifier[0].in_features
    hidden_layer = arg.hidden_units
    # get category names
    cat_to_name= load_json(arg.cat_names)
    output_unit = len(cat_to_name)
    epochs = arg.epochs
    print_every = arg.print_every
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_unit, hidden_layer[0])),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.1)),
        ('fc2', nn.Linear(hidden_layer[0], hidden_layer[1])),
        ('relu1', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.1)),
        ('fc3', nn.Linear(hidden_layer[1], output_unit)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=arg.lr)
                         
    return model, criterion, optimizer

def processor(gpu):
    if gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # if GPU fail, revert to CPU
    else:
        device = 'cpu'
    return device

# Save the checkpoint
def save_checkpoint(train_data, model, arg, optimizer):
    model.class_to_idx = train_data.class_to_idx
    torch.save({
            'classifier': model.classifier,
            'arch': arg.arch,                        # Saves architecture
            'state_dict': model.state_dict(),       # saves learnable params ONLY
            'class_to_idx': model.class_to_idx,
            'optimizer': optimizer.state_dict()
    }, arg.ckpt_dir)