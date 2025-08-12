
# conda activate ecoassistcondaenv-pytorch && python "C:\Users\smart\Desktop\deepfaune-training.py"

# test for running ds on C: or exSSD
# C:    - num_workers=0 EPOCH duration is ~ 10 min
# exSSD - num_workers=0 EPOCH duration is ~ 10 min
# exSSD - num_workers=2 EPOCH duration is ~ 3:40 min
# exSSD - num_workers=4 EPOCH duration is ~ 3:40 min

# path specification
dataset_dir_fpath = r"F:\datasets\2024-19-HWI-10K-seqdirs"
output_dir_fpath = r"C:\Peter\projects\2024-19-HWI"

# train settings
CROP_SIZE = 182
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 4e-4
PATIENCE = 10
BACKBONE = "vit_large_patch14_dinov2.lvd142m"

# import libraries
import os
import time
import numpy as np
import torch
from torch import nn, optim, tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import timm
from imgaug import augmenters as iaa # pip install imgaug
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import re
from pathlib import Path
import warnings
import csv
import pandas as pd
import random
from PIL import Image, ImageOps, ImageFile
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics # pip install scikit-learn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


# Init paths
train_dir_path = os.path.join(dataset_dir_fpath, "train")
test_dir_path = os.path.join(dataset_dir_fpath, "test")

# Configuration
CLASSES = next(os.walk(train_dir_path))[1]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### FUNCTIONS #####

# Define model with updated backbone
class Model(nn.Module):
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.base_model = timm.create_model(BACKBONE, pretrained=False, num_classes=NUM_CLASSES, dynamic_img_size=True)
        print(f"Using {BACKBONE} in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = NUM_CLASSES

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
        self.to(DEVICE)
        total_output = []
        with torch.no_grad():
            x = data.to(DEVICE)
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

        if path[-3:] != ".pt":
            path += ".pt"
        try:
            params = torch.load(path, map_location=DEVICE)
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

class Classifier:
    def __init__(self, weight_path):
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

# # Define model with updated backbone
# class CustomClassifier(nn.Module):
#     def __init__(self):
#         super(CustomClassifier, self).__init__()
#         self.base_model = timm.create_model(BACKBONE, pretrained=True, num_classes=NUM_CLASSES, dynamic_img_size=True)
    
#     def forward(self, x):
#         return self.base_model(x)

# # # Initialize model, loss, and optimizer
# # model = CustomClassifier()

# Set up data transformations using imgaug
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10), shear=(-5, 5)),
    iaa.Grayscale(alpha=(0.0, 1.0)),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),
    iaa.Multiply((0.8, 1.2)),
    iaa.LinearContrast((0.75, 1.5))
])

class ImgAugTransform:
    def __call__(self, img):
        img = np.array(img)
        img = augmenter(image=img)
        return img

def append_to_history_csv(history_csv, epoch, train_loss, train_acc, val_loss, val_acc):
    # Write or append history to CSV file
    file_exists = os.path.isfile(history_csv)
    with open(history_csv, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
    print(f"Training history saved to {history_csv}")

# plot training metrics
def plot_training_metrics(csv_file, session_dir):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract data from the DataFrame
    epochs_range = df['epoch']
    acc = df['train_acc']
    val_acc = df['val_acc']
    loss = df['train_loss']
    val_loss = df['val_loss']
    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(session_dir, 'training_metrics.png'))
    plt.close()


# plot images
def plot_sample_images(image_paths, output_file, title, grid_size=(4, 8)):
    num_images = grid_size[0] * grid_size[1]

    # Randomly sample or repeat images to match required number
    sampled_images = random.choices(image_paths, k=num_images)

    # Create the figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 2, grid_size[0] * 2))
    axes = axes.flatten()

    for ax, img_path in zip(axes, sampled_images):
        try:
            # Open and resize the image
            img = Image.open(img_path)
            img = img.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            ax.imshow(img)
            ax.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            ax.axis('off')

    # Add a title for the true class
    fig.suptitle(f"Class: {title}", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    print(f"Saving image grid to {output_file}...")
    plt.savefig(output_file)
    plt.close()

# Get a list of images from the specified directory
def get_image_list(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_list = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in image_extensions]
    return image_list

def evaluate_model(dataset_dir_fpath, session_dir):
    
    # log
    print("\nRunning class-specific validation on test set...")
    test_dir = os.path.join(dataset_dir_fpath, "test")
    print(f"test-set  : {test_dir}")

    # get device
    # # Check if the settings match
    # with open(os.path.join(session_dir, 'settings.json'), 'r') as f:
    #     saved_settings = json.load(f)
    
    # classes = saved_settings['CLASSES']
    # backbone = saved_settings['BACKBONE']
    # num_classes = saved_settings['NUM_CLASSES']
    # crop_size = saved_settings['CROP_SIZE']
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_fpath = os.path.join(session_dir, "best.pt")
    
    # get arguments
    params = torch.load(model_fpath, map_location=device)
    args = params['args']
    print(json.dumps(args, indent=2))
    classes = args['classes']
    global BACKBONE
    BACKBONE = args['backbone']
    global NUM_CLASSES
    NUM_CLASSES = args['num_classes']
    global CROP_SIZE
    CROP_SIZE = args['img_size']
    global DEVICE
    DEVICE = device
    

    
    
    
    # Export classification report to a text file
    eval_dir = os.path.join(session_dir, 'evaluation')
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    
    # exit()

    classifier = Classifier(weight_path= model_fpath)

    def get_classification(PIL_crop):
        PIL_crop = PIL_crop.convert('RGB')
        tensor_cropped = classifier.preprocessImage(PIL_crop)
        confs = classifier.predictOnBatch(tensor_cropped)[0,]
        lbls = classes
        classifications = []
        for i in range(len(confs)):
            classifications.append([lbls[i], confs[i]])
        return classifications

    # get subdir names which are the class names
    def get_classes(test_dir):
        return [name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))]
    print(f"get classes test dir : {get_classes(test_dir)}")

    # count
    total_imgs = 0
    for true_class in get_classes(test_dir):
        imgs = [os.path.join(test_dir, true_class, img) for img in os.listdir(os.path.join(test_dir, true_class)) if os.path.isfile(os.path.join(test_dir, true_class, img))]
        total_imgs += len(imgs)

    # Predict
    print("\nPredicting images on test set...\n")
    y_pred = []
    y_true = []
    wrong_classifications = defaultdict(list)  # Dictionary to store wrong classifications
    pbar = tqdm(total=total_imgs)

    for true_class in get_classes(test_dir):
        imgs = [os.path.join(test_dir, true_class, img) for img in os.listdir(os.path.join(test_dir, true_class)) if os.path.isfile(os.path.join(test_dir, true_class, img))]
        for img_path in imgs:
            img = Image.open(img_path)
            predictions = get_classification(img)
            pred_classes = max(predictions, key=lambda x: x[1])
            pred_class = pred_classes[0]
            y_pred.append(pred_class)
            y_true.append(true_class)

            # If the prediction is wrong, add to the wrong classifications dictionary
            if pred_class != true_class:
                wrong_classifications[true_class].append((img_path, pred_class))

            pbar.update(1)

    pbar.close()

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)


            

    # Print or save wrongly classified images by true class
    print("\nWrongly Classified Images:")
    for true_class, errors in wrong_classifications.items():
        print(f"\nTrue Class: {true_class}")
        output_file = os.path.join(eval_dir, f"{true_class.replace(' ', '-')}_misclassified.png")
        plot_wrong_predictions(errors, true_class, output_file)

                
    

    
    report_path = os.path.join(eval_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    threshold = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > threshold else "black")
    norm_cm_path = os.path.join(eval_dir, 'normalized_confusion_matrix.png')
    plt.savefig(norm_cm_path)
    plt.close()
    print(f"Normalized confusion matrix saved to {norm_cm_path}")

def plot_wrong_predictions(image_data, true_class, output_file, grid_size=(4, 8)):
    """
    Plot a grid of images with their predicted values and a title indicating the true class.

    Parameters:
    - image_data: List of tuples (image_path, predicted_class).
    - true_class: The true class label to display as the title.
    - output_file: File path to save the resulting image grid.
    - grid_size: Tuple indicating the grid dimensions (rows, columns).
    """
    num_images = grid_size[0] * grid_size[1]

    # Randomly sample or repeat images to match the required number
    sampled_data = random.choices(image_data, k=num_images)

    # Create the figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 2, grid_size[0] * 2))
    axes = axes.flatten()

    for ax, (img_path, pred_class) in zip(axes, sampled_data):
        try:
            # Open and resize the image
            img = Image.open(img_path)
            img = img.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            ax.imshow(img)
            ax.set_title(pred_class, fontsize=12)  # Show the predicted class
            ax.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            ax.axis('off')

    # Add a title for the true class
    fig.suptitle(f"True class: {true_class}", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    print(f"Saving image grid to {output_file}...")
    plt.savefig(output_file)
    plt.close()



# check the highest subdirectory index when the pattern is f'train{subdir_index}'
def fetch_session_dir():
    resume = False
    
    # list dirs
    pytorch_dir = os.path.join(output_dir_fpath, 'pt-self')
    Path(pytorch_dir).mkdir(parents=True, exist_ok=True)
    subdirs = [d for d in os.listdir(pytorch_dir) if os.path.isdir(os.path.join(pytorch_dir, d))]
    train_dirs = [d for d in subdirs if re.match(r"train\d+$", d)]
    
    # check if there have been pervious sessions
    if not train_dirs:
        # there are no previous sessions
        # create the first session subdir 
        print("There are no previous sessions")
        session_dir = os.path.join(pytorch_dir, 'train1')
        resume = False

        
    else:
        # there are previous sessions
        # check the latest session subdir
        prev_session_dir = max(train_dirs, key=lambda d: int(re.search(r"\d+$", d).group()))
        prev_session_idx = int(re.search(r"\d+$", prev_session_dir).group())
        
        # check if there is a checkpoint file in the latest session subdir
        checkpoint_path = os.path.join(pytorch_dir, prev_session_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            
            print("There is a checkpoint file present in previous session")
            resume_from_checkpoint = input("Do you want to resume from the last checkpoint? (yes/no): ").strip().lower()
            if resume_from_checkpoint == 'yes' or resume_from_checkpoint == 'y':
                session_dir = os.path.join(pytorch_dir, prev_session_dir)
                resume = True
            elif resume_from_checkpoint == 'no' or resume_from_checkpoint == 'n':
                session_dir = os.path.join(pytorch_dir, f'train{prev_session_idx + 1}')
                resume = False
                
        else:
            print("There is no checkpoint file present in previous session")
            session_dir = os.path.join(pytorch_dir, f'train{prev_session_idx + 1}')
            resume = False
            
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    print(f"session dir is set to {session_dir}")
    return session_dir, resume
            

def train():

    # check previous sessions
    session_dir, resume_from_previous_session = fetch_session_dir()

    # change directory to session dir
    os.chdir(session_dir) # TODO: make use of obsolute paths only. That will make multiple disk use possible. Perhaps first try it, but then you'll know what the problem is. 



    # Export settings to JSON
    settings_path = os.path.join(session_dir, 'settings.json')
    if not os.path.exists(settings_path):
        with open(settings_path, 'w') as f:
            json.dump({
                "BATCH_SIZE": BATCH_SIZE,
                "CROP_SIZE": CROP_SIZE,
                "EPOCHS": EPOCHS,
                "PATIENCE": PATIENCE,
                "BACKBONE": BACKBONE,
                "CLASSES": CLASSES,
                "NUM_CLASSES": NUM_CLASSES,
                "TRAIN_SET_PATH": dataset_dir_fpath
            }, f, indent=2)
        print(f"Settings saved to {settings_path}")
    else:
        print(f"Settings already exist at {settings_path}")



    if not resume_from_previous_session:
        class_dirs = os.listdir(os.path.join(dataset_dir_fpath, "train"))
        for class_dir in class_dirs:
            # get list of images
            class_train_dir_path = os.path.join(dataset_dir_fpath, "train", class_dir)
            class_test_dir_path = os.path.join(dataset_dir_fpath, "test", class_dir)
            class_train_image_list = get_image_list(class_train_dir_path)
            class_test_image_list = get_image_list(class_test_dir_path)
            save_dir = os.path.join(session_dir, 'random-subsamples', class_dir)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plot_sample_images(class_train_image_list, os.path.join(save_dir, 'train.png'), class_dir)
            plot_sample_images(class_train_image_list, os.path.join(save_dir, 'test.png'), class_dir)

    # suppress warning about flash attention
    warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")


    train_transforms = transforms.Compose([
        transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
    ])




    # Load datasets
    train_data = datasets.ImageFolder(train_dir_path, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir_path, transform=test_transforms)

    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)




    # Initialize model, loss, and optimizer
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # Check if a checkpoint exists
    checkpoint_path = os.path.join(session_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        
        # init
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Check if the settings match
        with open(os.path.join(session_dir, 'settings.json'), 'r') as f:
            saved_settings = json.load(f)
        
        if (saved_settings['BATCH_SIZE'] == BATCH_SIZE and
            saved_settings['CROP_SIZE'] == CROP_SIZE and
            saved_settings['EPOCHS'] == EPOCHS and
            saved_settings['PATIENCE'] == PATIENCE and
            saved_settings['BACKBONE'] == BACKBONE and
            saved_settings['CLASSES'] == CLASSES and
            saved_settings['NUM_CLASSES'] == NUM_CLASSES and
            saved_settings['TRAIN_SET_PATH'] == dataset_dir_fpath):
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Saved settings:")
            for key, value in saved_settings.items():
                print(f"{key}: {value}")
            
            print("Current settings:")
            current_settings = {
                "BATCH_SIZE": BATCH_SIZE,
                "CROP_SIZE": CROP_SIZE,
                "EPOCHS": EPOCHS,
                "PATIENCE": PATIENCE,
                "BACKBONE": BACKBONE,
                "CLASSES": CLASSES,
                "NUM_CLASSES": NUM_CLASSES,
                "TRAIN_SET_PATH": dataset_dir_fpath
            }
            for key, value in current_settings.items():
                print(f"{key}: {value}")
                
            raise ValueError("Settings do not match. Make sure you are running the script the same settings as the saved checkpoint. Or retry and start new training.")
    else:
        start_epoch = 0

    # Check for available device
    model.to(DEVICE)
    print(f"sending mdoel to ... {DEVICE}")

    # ensure the optimizer’s state is moved to the correct device
    for state in optimizer.state.values():
        if isinstance(state, torch.Tensor):
            state.data = state.data.to(DEVICE)
        elif isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(DEVICE)

    # Training with early stopping
    best_val_accuracy = 0
    patience_counter = 0

    # code to save training history
    history_csv = os.path.join(session_dir, 'training_history.csv')


    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=session_dir)
    print(f"TensorBoard writer initialized")
    print(f"\n\nto view the training logs, run:\n")
    print(f"conda activate ecoassistcondaenv-pytorch && tensorboard --logdir={os.path.join(output_dir_fpath, 'pt-self')} \n\n")
    # conda activate ecoassistcondaenv-pytorch && tensorboard --logdir=C:\Peter\projects\2024-19-HWI\runs

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for images, labels in train_progress:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            train_progress.set_postfix(loss=(running_loss / len(train_loader)))
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        # Evaluation on validation set for early stopping
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = correct / total
        
        # Log validation metrics
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        # Write history to CSV file
        append_to_history_csv(history_csv, epoch, train_loss, train_acc, val_loss, val_acc)
        
        # update plots
        plot_training_metrics(history_csv, session_dir)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Early stopping logic
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'args': {
                    'backbone': BACKBONE,
                    'num_classes': NUM_CLASSES,
                    'classes': CLASSES,
                    'img_size': CROP_SIZE,
                }
            }, "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved")

    # Final model saving
    torch.save({'state_dict': model.state_dict(),
                'args': {
                    'backbone': BACKBONE,
                    'num_classes': NUM_CLASSES,
                    'classes': CLASSES,
                    'img_size': CROP_SIZE,
                }
            }, "last.pt")

    # Close the writer
    writer.close()


    # remove the checkpoint file if the training successfully completes
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed")




    # evaluate model
    evaluate_model(dataset_dir_fpath, session_dir)



if __name__ == '__main__':
    train()


