#import json
import torch
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#from PIL import Image

from util_functions import input_args#, load_json
from model_functions import load_dataset, build_classifier, processor, save_checkpoint

arg = input_args()
print(arg)

data_dir = arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_data, trainloader, validloader, testloader = load_dataset(train_dir, valid_dir, test_dir)  #train_data needed for saving checkpoint
    
model, criterion, optimizer = build_classifier(arg)
print(model)

device = processor(arg.gpu)

def validation(model, validloader, criterion):
    loss = 0
    valid_loss = 0
    accuracy = 0
    for ii, (images, labels) in enumerate(validloader):
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        valid_loss += criterion(output, labels)

        ps = torch.exp(output)  # suggested to use with NLLLoss
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

model.to(device)
model.train()
steps = 0

print('Training...')
for epoch in range(arg.epochs):
    running_loss = 0
    loss = 0
    for ii, (images, labels) in enumerate(trainloader):
        steps += 1
        images, labels = images.to(device), labels.to(device)

        # clear gradients
        optimizer.zero_grad()

        # forward and backward passes
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #if steps % 50 == 0: #print every 50 images
            #print('# of images trained is: {}'.format(steps))

    if steps % arg.print_every == 0:
        # evaluate the valid dataset
        model.eval()
        print('Validating...')

        with torch.no_grad():
            valid_loss, accuracy = validation(model, validloader, criterion)

        print("Epoch: {}/{}.. ".format(epoch + 1, arg.epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / arg.print_every),
              "Valid Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
              "Valid Accuracy: {:.3f}".format(accuracy / len(validloader)))
        running_loss = 0
        model.train()       #revert model to train mode

# Test model Accuracy
correct_class = 0
total = 0
print('Testing...')
with torch.no_grad():
    for ii, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct_class += (predicted == labels).sum().item()

print('Accuracy for test images: %d %%' % (correct_class / total * 100))
    
save_checkpoint(train_data, model, arg, optimizer)