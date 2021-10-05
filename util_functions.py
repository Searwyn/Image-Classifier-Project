import json
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms

def input_args():
    """
    Retrieves and parses command line arguments provided by the user from a terminal window.

    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments objects
    """

    # Creates Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Creates arguments
    parser.add_argument('--data_dir', type=str, default='flowers', help='filepath to dataset')
    parser.add_argument('--cat_names', type=str, default='cat_to_name.json',
                        help='index mapping to real names in json file')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN architecture (vgg16 [default], alexnet, resnet)')
    parser.add_argument('--gpu', type=bool, default=True, help='Processor (True - GPU [default], False - CPU)')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoint_v1.pth', help='checkpoints directory')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='# of epochs')
    parser.add_argument('--print_every', type=int, default=5, help='print values every number of steps')
    parser.add_argument('--hidden_units', type=list, default=[1024, 512],
                        help='# of nodes per layer in a list format: [a, b, c, ...]')
    #parser.add_argument('--img_path', type=str, default='flowers/test/1/image_06743.jpg',
                        #help='image path for testing')
    parser.add_argument('--img_path', type=str, default='flowers/test/11/image_03098.jpg',
                        help='image path for testing')
    parser.add_argument('--top_k', type=int, default=3, help='# of most likely classes')
    parser.add_argument('--show', type=bool, default=False, help='show plot (False [default])')

    # parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'path to the data folder')
    # parser.add_argument('--arch', type = str, default = 'vgg', help = 'CNN model architecture (resnet, alexnet, vgg)')
    # parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
    # parser.add_argument('--hidden_units', type = list, default = [1024,512] , help = '# of nodes per layer in list format: [a, b, c, ...]')
    # parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs')
    # parser.add_argument('--print_every', type = int, default = 60, help = 'print values every number of steps')
    # parser.add_argument('--gpu', type = bool, default = True, help = 'Processor selection (True - GPU (default), False - CPU)')
    # parser.add_argument('--cp_dir', type = str, default = 'checkpoint.pth', help = 'checkpoints directory')
    # parser.add_argument('--img_path', type = str, default = 'flowers/test/1/image_06743.jpg', help = 'image path')
    # parser.add_argument('--top_k', type = int, default = 3, help = 'number of most likely classes')
    # parser.add_argument('--cat_names', type = str, default = 'cat_to_name.json', help = 'mapping of categories to real names - json file')

    return parser.parse_args()

# load json file
def load_json(filepath):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

# a function that loads a checkpoint and rebuilds the model
def checkpoint(filepath):
    # to allow the model trained on GPU to be used on CPU or vice versa.
    checkpoint = torch.load((filepath), map_location=lambda storage, loc: storage)
    model= getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = checkpoint['optimizer']
    
    num_labels = len(checkpoint['class_to_idx'])
    #print("State dict keys:\n", model.state_dict().keys())

    return model

def process_image(image):
    pic= Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    pic_trnasform= transform(pic)
    return pic_trnasform

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if device == torch.device('cuda:0'):
        print('Currently in GPU Mode')
        image = process_image(image_path).type(torch.cuda.FloatTensor).unsqueeze_(0)
    else:
        print('Currently in Cpu Mode')
        image = process_image(image_path).type(torch.FloatTensor).unsqueeze_(0)

    image = image.to(device)    #to prevent making a new tensor
    model = checkpoint(model)
    model = model.to(device)
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    model.eval()
    
    with torch.no_grad():
        output = model.forward(image)
        ps= torch.exp(output)    #suggested to use with NLLLoss 
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()

        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
            
    return probs, classes

# Display an image along with the top k classes
def plot_predict(filepath, model):
    # create two figures
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (6,10))

    # plot image
    cat_to_name= load_json(arg.cat_names)
    img_name = [cat_to_name[filepath.split('/')[-2]]]
    imshow(process_image(filepath), ax1, title = img_name)
    
    
    ax1.axis('off')
    ax1.set_title(cat_to_name[filepath.split('/')[2]])  #Title of figure

    # plot prediction
    probs, classes = predict(filepath, model)
    flower_names = [cat_to_name[k] for k in classes]    #get names based on class number

    print('Top 5 Probabilities: ',probs, '\nTop 5 Classes: ', classes)
    print('Top 5 Flower Names: ', flower_names)
    
    ax2.barh(classes, probs, align = 'center', color = 'darkblue')
    ax2.set_aspect(0.2)
    ax2.set_yticks(classes) #set y axis tick locations
    ax2.set_yticklabels(flower_names)           # class name
    ax2.set_title('Class Probability')
    ax2.set_xlabel('Probability')               # axis name
    ax2.set_xlim(0,1.1)                         #x axis limits
    
    plt.tight_layout()
    
    return ax1, ax2