import numpy as np
import torch,os,random
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms#, models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.utils.model_zoo as model_zoo
from model_core import Two_Stream_Net

import os,glob
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from loss.am_softmax import AMSoftmaxLoss
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm


logging.basicConfig(filename='training_FS.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Model: Two_Stream_Net + Gram")

torch.nn.Module.dump_patches = True

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

root='../'

def default_loader(path):

    # size = random.randint(64, 256)

    im=cv2.imread(path)
    if im is None:
        print("Failed to load image:", path)
        logging.info("Failed to load image: " + path)
        return None
    # im=cv2.resize(im,(size,size))
    # im=cv2.resize(im,(256,256))
    ims=np.zeros((3,256,256))
    ims[0,:,:]=im[:,:,0]
    ims[1,:,:]=im[:,:,1]
    ims[2,:,:]=im[:,:,2]
    img_tensor=torch.tensor(ims.astype('float32'))
    
    return img_tensor

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        if img is None:
            # Return a placeholder tensor instead of None
            return torch.zeros(3, 256, 256), -1  # Placeholder tensor and a dummy label

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

image_datasets = customData(img_path='',txt_path=('1train_FS.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

dataloders =  torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=4,
                                                 shuffle=True) 

test_datasets = customData(img_path='',txt_path=('1val_FS.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False) 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Two_Stream_Net()

criterion = AMSoftmaxLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
model.to(device)



def test(model, testloader, criterion):
    model.eval()
    predictions = []
    targets = []
    correct = 0
    total = 0
    loss_sum = 0

    for inputs, labels in testloader:
        # skip -1 labels
        if -1 in labels:
            print("Skipping batch due to missing data")
            continue # 跳过当前 batch，继续下一个 batch 的训练

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs, _, _ = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        # Collect predictions and ground truth labels
        predictions.extend(probs[:, 1].detach().cpu().numpy())  # Assuming class 1 is the positive class
        targets.extend(labels.detach().cpu().numpy())

        # Compute accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Compute loss
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * inputs.size(0)

    accuracy = correct / total
    auc = roc_auc_score(targets, predictions)
    avg_loss = loss_sum / len(testloader.dataset)
    return accuracy, auc, avg_loss


lr=0.00001
for param_group in optimizer.param_groups:
  param_group['lr']=lr


def get_lr(optimizer):
  lr=[]
  for param_group in optimizer.param_groups:
    lr+=[param_group['lr']]
  return lr


epochs = 10
steps = 0
running_loss = 0
#pri_every = 1000
train_losses, val_losses = [], []


# eraly stopping parameters
best_val_auc = 0.0
best_model_state = None
patience = 3
no_improvement_count = 0

for epoch in tqdm(range(epochs), desc='Training Progress'):
    current_lr = get_lr(optimizer)
    logging.info(f"Current learning rate: {current_lr}")

    running_loss = 0.0

    for inputs, labels in tqdm(dataloders, desc='Epoch {} Progress'.format(epoch)):
        # skip -1 labels
        if -1 in labels:
            print("Skipping batch due to missing data")
            continue 

        model.train()
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps[0], labels)
        #print(epoch, 'loss', loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # loss of the epoch
    epoch_loss = running_loss / len(dataloders)
    train_losses.append(epoch_loss)

    # validation and calculate AUC
    model.eval()
    with torch.no_grad():
        val_acc, val_auc, val_loss = test(model, testloader, criterion)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val AUC: {val_auc:.4f}")

    # early stopping: if no improvement in validation AUC, increment counter; if improved, reset counter
    if val_auc <= best_val_auc:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            logging.info(f"No improvement in validation AUC for {patience} epochs. Early stopping...")
            break
    else:
        best_val_auc = val_auc
        best_model_state = model.state_dict()
        no_improvement_count = 0
        # use the best model parameters on the validation set
        torch.save(best_model_state, 'gram_best_model_FS.pth')
        logging.info("best model saved.")

    logging.info(f"Epoch [{epoch + 1}/{epochs}], Step: {steps}") # , Loss: {loss.item():.4f}
    logging.info(
        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")


plt.plot(train_losses, label='Training Loss')
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(root + 'img/gram_loss_curve_FS.png')

logging.shutdown()
