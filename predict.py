from utils import *

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

parser = argparse.ArgumentParser()

parser.add_argument('--topk', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use mapping of categories to real names')
parser.add_argument('--gpu', default='gpu', type=str, help='Use GPU for inference')
parser.add_argument('--input_image', type=str, default='./flowers/test/1/image_06743.jpg', help='Input image for prediction')
parser.add_argument('--checkpoint', type=str, default='model.pth', help='Load saved checkpoint/model')

# Client input
client_input = parser.parse_args()
topk = client_input.topk
category_names = client_input.category_names
gpu = client_input.gpu
input_image = client_input.input_image
checkpoint = client_input.checkpoint

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    class_to_idx = checkpoint['class_to_idx']

    return model.eval()

# Load a checkpoint
path = 'model.pth'
model = load_checkpoint(path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Process PIL image for a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    width = 256
    height = 256
    img = Image.open(image).resize((width, height))
    
    process = transforms.Compose([transforms.CenterCrop(224), 
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
    img = process(img)
    
    return img

# Function to predict the class from an image file
def predict(image_path, model, topk=topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img.to(device))
        
    probability = torch.exp(logps)
    
    probs, classes = probability.topk(topk)
    probs = probs.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    
    class_to_idx = {value:key for key, value in model.class_to_idx.items()}
    classes = [class_to_idx[idx] for idx in classes]
    
    return probs, classes

# Make an inference
probs, classes = predict(input_image, model)
print('Predicted Probabilities:\n', probs)
print('Predicted Classes:\n', classes)