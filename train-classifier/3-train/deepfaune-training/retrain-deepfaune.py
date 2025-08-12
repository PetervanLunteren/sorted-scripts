
# conda activate ecoassistcondaenv-pytorch && python "C:\Users\smart\Desktop\retrain-deepfaune.py"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
import os
import sys
import numpy as np
import timm
import torch
from torch import tensor
import torch.nn as nn
from torchvision.transforms import InterpolationMode, transforms

# Init paths
dataset_dir_fpath = r"F:\datasets\2024-19-HWI-10K"
train_dir_path = os.path.join(dataset_dir_fpath, "train")
test_dir_path = os.path.join(dataset_dir_fpath, "test")

# Configuration
CLASSES = next(os.walk(train_dir_path))[1]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CROP_SIZE = 182
BACKBONE = "vit_large_patch14_dinov2.lvd142m"
weight_path = r'C:\Users\smart\Desktop\cls-models\published\Europe - DeepFaune v1.1\deepfaune-vit_large_patch14_dinov2.lvd142m.pt'
# NUM_CLASSES = 10

txt_animalclasses = {
    'fr': ["blaireau", "bouquetin", "cerf", "chamois", "chat", "chevre", "chevreuil", "chien", "ecureuil", "equide", "genette",
           "herisson", "lagomorphe", "loup", "lynx", "marmotte", "micromammifere", "mouflon",
           "mouton", "mustelide", "oiseau", "ours", "ragondin", "renard", "sanglier", "vache"],
    'en': ["badger", "ibex", "red deer", "chamois", "cat", "goat", "roe deer", "dog", "squirrel", "equid", "genet",
           "hedgehog", "lagomorph", "wolf", "lynx", "marmot", "micromammal", "mouflon",
           "sheep", "mustelid", "bird", "bear", "nutria", "fox", "wild boar", "cow"],
    'it': ["tasso", "stambecco", "cervo", "camoscio", "gatto", "capra", "capriolo", "cane", "scoiattolo", "equide", "genet",
           "riccio", "lagomorfo", "lupo", "lince", "marmotta", "micromammifero", "muflone",
           "pecora", "mustelide", "uccello", "orso", "nutria", "volpe", "cinghiale", "mucca"],
    'de': ["Dachs", "Steinbock", "Rothirsch", "Gämse", "Katze", "Ziege", "Rehwild", "Hund", "Eichhörnchen", "Equiden", "Ginsterkatze",
           "Igel", "Lagomorpha", "Wolf", "Luchs", "Murmeltier", "Kleinsäuger", "Mufflon",
           "Schaf", "Mustelide", "Vogen", "Bär", "Nutria", "Fuchs", "Wildschwein", "Kuh"],
    
}

class Classifier:
    def __init__(self):
        self.model = Model()
        self.model.loadWeights(weight_path)
        self.transforms = transforms.Compose([
            transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))])

    def predictOnBatch(self, batchtensor, withsoftmax=True):
        return self.model.predict(batchtensor, withsoftmax)

    # croppedimage loaded by PIL
    def preprocessImage(self, croppedimage):
        preprocessimage = self.transforms(croppedimage)
        return preprocessimage.unsqueeze(dim=0)

class Model(nn.Module):
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.base_model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(txt_animalclasses['fr']),
                                            dynamic_img_size=True)
        print(f"Using {BACKBONE} with weights at {weight_path}, in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = len(txt_animalclasses['fr'])

    def forward(self, input):
        x = self.base_model(input)
        return x

    def predict(self, data, withsoftmax=True):
        """
        Predict on test DataLoader
        :param test_loader: test dataloader: torch.utils.data.DataLoader
        :return: numpy array of predictions without soft max
        """
        self.eval()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = DEVICE
        self.to(device)
        total_output = []
        with torch.no_grad():
            x = data.to(device)
            if withsoftmax:
                output = self.forward(x).softmax(dim=1)
            else:
                output = self.forward(x)
            total_output += output.tolist()

        return np.array(total_output)

    def loadWeights(self, path):
        """
        :param path: path of .pt save of model
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = DEVICE

        if path[-3:] != ".pt":
            path += ".pt"
        try:
            params = torch.load(path, map_location=device)
            args = params['args']
            if self.nbclasses != args['num_classes']:
                raise Exception("You load a model ({}) that does not have the same number of class"
                                "({})".format(args['num_classes'], self.nbclasses))
            self.backbone = args['backbone']
            self.nbclasses = args['num_classes']
            self.load_state_dict(params['state_dict'])
        except Exception as e:
            print("\n/!\ Can't load checkpoint model /!\ because :\n\n " + str(e), file=sys.stderr)
            raise e



# STEP 1

import torch
import torch.nn as nn

# # Load the pre-trained model
# original_model = torch.load('original_model.pth', map_location='cpu')

# # Move the model to the appropriate device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# original_model.to(device)



original_model = Classifier()



# STEP 2

num_new_classes = NUM_CLASSES 

# Assuming the model has a fully connected layer named `fc` as the last layer
in_features = original_model.fc.in_features
original_model.fc = nn.Linear(in_features, num_new_classes)
original_model.to(device)

# Initialize the new layer
nn.init.xavier_uniform_(original_model.fc.weight)
nn.init.zeros_(original_model.fc.bias)


exit()


# Load the pre-trained model
# model = timm.create_model(BACKBONE, pretrained=False, num_classes=NUM_CLASSES, dynamic_img_size=True)
# model.load_state_dict(torch.load(weight_path))


model = Model()
model.loadWeights(weight_path)
transforms = transforms.Compose([
            transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))])


# model = Classifier()

# Modify the final layer to match the number of new classes
num_new_classes = NUM_CLASSES  # Replace with the actual number of new classes
# model.head = nn.Linear(model.head.in_features, num_new_classes)

model.base_model.head = nn.Linear(model.base_model.head.in_features, num_new_classes)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((182, 182)),
    transforms.ToTensor(),
])

# Load datasets
train_data = datasets.ImageFolder(r"F:\datasets\2024-19-HWI-10K\train", transform=transform)
val_data = datasets.ImageFolder(r"F:\datasets\2024-19-HWI-10K\test", transform=transform)

# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed
model.train()

print('Starting training...')

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}%')
    model.train()

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_model.pt')