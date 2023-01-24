from utils import *

# Load data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# the validation transforms
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# training dataset
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

# test_dataset
test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

# validation dataset
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# training data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# test data loaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# validation data loaders
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Set directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Choose network architecture')
# Set hyperparameters
parser.add_argument('--data_dir', type=str, default="./flowers/", help='Set directory for loading data')
parser.add_argument('--learning_rate', type=float, default=.001, help='Set learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='Assign value for the hidden layer')
parser.add_argument('--epochs', type=int, default=5, help='Choose number of epochs to train the model')
parser.add_argument('--gpu', default='gpu', type=str, help='Use GPU for training')

# Client Input
client_input = parser.parse_args()
save_dir = client_input.save_dir
arch = client_input.arch
data_dir = client_input.data_dir
lr = client_input.learning_rate
hidden_units = client_input.hidden_units
epochs = client_input.epochs
gpu = client_input.gpu

# Functions and classes relating to the model
def Classifier(arch='vgg16', hidden_units=2048):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
                        nn.Linear(input_size, hidden_units), 
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
    
                        nn.Linear(hidden_units, 256),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                                 
                        nn.Linear(256, 102),
                        nn.LogSoftmax(dim=1))

    model.classifier = classifier

    return model

model = Classifier(arch=arch, hidden_units=hidden_units)
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)


def train_model(model, epochs, train_loader, valid_loader, lr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model = model.to(device)
    
    print_every = 5
    steps = 0
    loss_show = []

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

# Train model
train_model(model, epochs, train_loader, valid_loader, lr)

def test_accuracy(model):
    test_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(test_loader):.3f}")

# Check Accuracy of model    
test_accuracy(model)

# Save Checkpoint
image_datasets = {'train': train_data,
                  'valid': valid_data,
                  'test': test_data}

model.class_to_idx = image_datasets['train'].class_to_idx
EPOCH = epochs
PATH = "model.pth"

torch.save({
            'epoch': EPOCH,
            'model': model,
            'class_to_idx': model.class_to_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)