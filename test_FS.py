import torch, os, random
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from model_core_gram import Two_Stream_Net
from tqdm import tqdm

def default_loader(path):

    # size = random.randint(64, 256)

    im=cv2.imread(path)
    if im is None:
        print("Failed to load image:", path)
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
 

def test(model, testloader):
    model.eval()
    predictions = []
    targets = []
    correct = 0
    total = 0

    for inputs, labels in tqdm(testloader, desc="Testing"):
        if -1 in labels:
            print("Skipping batch due to missing data")
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _, _ = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        predictions.extend(probs[:, 1].detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    auc = roc_auc_score(targets, predictions)
    precision = precision_score(targets, np.round(predictions))
    recall = recall_score(targets, np.round(predictions))
    f1 = f1_score(targets, np.round(predictions))

    return accuracy, auc, precision, recall, f1, predictions, targets

## FS_DF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelinit = torch.load('gram_best_model_FS.pth')
model = Two_Stream_Net()
model.load_state_dict(modelinit)
model.to(device)

test_datasets = customData(img_path='',txt_path=('test_DF.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False) 

print('FS | DF')
acc, auc, precision, recall, f1, predictions, targets = test(model, testloader)

fw = open('result.txt', 'a')
fw.write('Train: FS | Test: DF | acc: {} auc: {} precision: {} recall: {} f1: {}\n'.format(acc, auc, precision, recall, f1))
fw.flush()


## FS_F2F
test_datasets = customData(img_path='',txt_path=('test_F2F.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False) 

print('FS | F2F')
acc, auc, precision, recall, f1, predictions, targets = test(model, testloader)

fw = open('result.txt', 'a')
fw.write('Train: FS | Test: F2F | acc: {} auc: {} precision: {} recall: {} f1: {}\n'.format(acc, auc, precision, recall, f1))
fw.flush()


## FS_FS
test_datasets = customData(img_path='',txt_path=('test_FS.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False) 

print('FS | FS')
acc, auc, precision, recall, f1, predictions, targets = test(model, testloader)

fw = open('result.txt', 'a')
fw.write('Train: FS | Test: FS | acc: {} auc: {} precision: {} recall: {} f1: {}\n'.format(acc, auc, precision, recall, f1))
fw.flush()


## FS_NT
test_datasets = customData(img_path='',txt_path=('test_NT.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False) 

print('FS | NT')
acc, auc, precision, recall, f1, predictions, targets = test(model, testloader)

fw = open('result.txt', 'a')
fw.write('Train: FS | Test: NT | acc: {} auc: {} precision: {} recall: {} f1: {}\n'.format(acc, auc, precision, recall, f1))
fw.flush()
