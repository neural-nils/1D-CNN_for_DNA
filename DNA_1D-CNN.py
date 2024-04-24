


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import torch.nn.functional as F

dolphin_reads = open("SRR25750764.dolphin.txt", "r").read().splitlines()
human_reads = open("SRR9091899.human.txt", "r").read().splitlines()
bat_reads = open("SRR18788643.myotis.txt", "r").read().splitlines()

max_seq_len = max([len(seq) for seq in dolphin_reads]+[len(seq) for seq in human_reads]+[len(seq) for seq in bat_reads])
min_seq_len = min([len(seq) for seq in dolphin_reads]+[len(seq) for seq in human_reads]+[len(seq) for seq in bat_reads])

print("trimming reads to uniform size...")       
dolphin_reads = [s[2:min_seq_len-2] for s in dolphin_reads]
human_reads = [s[2:min_seq_len-2] for s in human_reads]
bat_reads = [s[2:min_seq_len-2] for s in bat_reads]


print("length distribution in the 3 samples:")
print(set([len(seq) for seq in dolphin_reads]))
print(set([len(seq) for seq in human_reads]))
print(set([len(seq) for seq in bat_reads]))

print("obtained reads of each sample:")        
print(len(dolphin_reads))
print(len(human_reads))
print(len(bat_reads))

encoder = {"A" : 0, "T" : 1, "C" : 2, "G" : 3 }
decoder = {v:k for k,v in encoder.items()}

print("start one-hot encoding...")
def one_hot_encode(seq, nt_to_idx):    
    seq = np.array([nt_to_idx[c] for c in seq])
    one_hot = np.zeros((seq.size, 4))
    one_hot[np.arange(seq.size), seq] = 1        
    return torch.tensor(one_hot, dtype=torch.float32)
    
sequences_encoded_dolphin = [one_hot_encode(seq, encoder) for seq in dolphin_reads if 'N' not in seq]
sequences_encoded_human = [one_hot_encode(seq, encoder) for seq in human_reads if 'N' not in seq]
sequences_encoded_bat = [one_hot_encode(seq, encoder) for seq in bat_reads if 'N' not in seq]


print(f"{len(sequences_encoded_dolphin)} dolphin sequences")
print(f"{len(sequences_encoded_human)} human sequences")
print(f"{len(sequences_encoded_bat)} bat sequences")


train_dolphin_num = int(np.floor(len(sequences_encoded_dolphin) * 0.8))
test_dolphin_num = len(sequences_encoded_dolphin) - train_dolphin_num

train_human_num = int(np.floor(len(sequences_encoded_human) * 0.8))
test_human_num = len(sequences_encoded_human) - train_human_num

train_bat_num = int(np.floor(len(sequences_encoded_bat) * 0.8))
test_bat_num = len(sequences_encoded_bat) - train_bat_num


#TRAIN
sequences_encoded_dolphin_train = sequences_encoded_dolphin[:train_dolphin_num] 
sequences_encoded_human_train = sequences_encoded_human[:train_human_num]
sequences_encoded_bat_train = sequences_encoded_bat[:train_bat_num]

sequences_encoded_train = sequences_encoded_bat_train + sequences_encoded_dolphin_train + sequences_encoded_human_train 
sequences_padded_train = nn.utils.rnn.pad_sequence(sequences_encoded_train, batch_first=True)

label_dict = {2 : "bat", 1 : "dolphin", 0 : "human"}
labels_train = [2]*len(sequences_encoded_bat_train) + [1]*len(sequences_encoded_dolphin_train) + [0]*len(sequences_encoded_human_train)
labels_tensor_train = torch.tensor(labels_train, dtype=torch.long)

#TEST
sequences_encoded_dolphin_test = sequences_encoded_dolphin[train_dolphin_num:] 
sequences_encoded_human_test = sequences_encoded_human[train_human_num:]
sequences_encoded_bat_test = sequences_encoded_bat[train_bat_num:]

sequences_encoded_test = sequences_encoded_bat_test + sequences_encoded_dolphin_test + sequences_encoded_human_test
sequences_padded_test = nn.utils.rnn.pad_sequence(sequences_encoded_test, batch_first=True)
labels_test = [2]*len(sequences_encoded_bat_test) + [1]*len(sequences_encoded_dolphin_test) + [0]*len(sequences_encoded_human_test)
labels_tensor_test = torch.tensor(labels_test, dtype=torch.long)


print("assembling dataset")
dataset_train = TensorDataset(sequences_padded_train, labels_tensor_train)
dataset_test = TensorDataset(sequences_padded_test, labels_tensor_test)
    
dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)


class DNA_CNN3(nn.Module):
    def __init__(self, max_length=150):
        super(DNA_CNN3, self).__init__()
        self.max_length = max_length        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=1)        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 15, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):        
        x = self.pool(nn.functional.relu(self.conv1(x)))                                
        x = self.pool(nn.functional.relu(self.conv2(x)))                                
        x = self.pool(nn.functional.relu(self.conv3(x)))                    
        x = x.view(-1, 128 * 15)                                                
        x = self.dropout(nn.functional.relu(self.fc1(x)))                        
        x = self.dropout(nn.functional.relu(self.fc2(x)))            
        x = self.dropout(nn.functional.relu(self.fc3(x)))            
        x = self.fc4(x)        
        return x
    
    
max_sequence_length_dolphin = max(len(seq) for seq in sequences_encoded_dolphin)
max_sequence_length_human = max(len(seq) for seq in sequences_encoded_human)
max_sequence_length_bat = max(len(seq) for seq in sequences_encoded_bat)

max_sequence_length = max(max_sequence_length_dolphin, max_sequence_length_human, max_sequence_length_bat)

print("max_sequence_length",max_sequence_length)

model = DNA_CNN3(max_length=max_sequence_length).cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    
step_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1) 
#step_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 


print("start training...")    
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("cuda available... utilizing GPU...")        

    
num_epochs = 20
losses = []    

for epoch in range(num_epochs):
    model.train()
    loss_sum = 0
    count = 0    
    
    for inputs, labels in dataloader:
        
        inputs = inputs.permute(0,2,1)        
        inputs, labels = inputs.to(device), labels.to(device)                        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_sum += loss            
        count += 1            
        
        loss.backward()                    
        optimizer.step()
                        
    avg_loss = loss_sum/count        
    losses.append(avg_loss)        
    step_scheduler.step()
    
    print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss}')
        
    #output train accuracy every 5 epochs:
    if epoch % 5 == 0:        
        model.eval()
        total_num = 0
        total_correct = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:            
                inputs = inputs.permute(0,2,1)            
                inputs, labels = inputs.to(device), labels.to(device)            
                pred = model(inputs)                                                
                pred = F.softmax(pred, dim=1)
                predicted_labels = np.argmax(pred.detach().cpu().numpy(), axis=1)
                labels = labels.cpu().numpy()
                correct = np.sum(labels == predicted_labels)                
                num = len(predicted_labels)
                total_num += num
                total_correct += correct                    
                accuracy = total_correct/total_num
        print(f"train accuracy after epoch {epoch}: total_correct={total_correct}, total_num={total_num} accuracy={total_correct/total_num}")
        

print("checking test accuracy...")
model.eval()
total_num = 0
total_correct = 0

with torch.no_grad():
    for inputs, labels in dataloader_test:                        
        
        inputs = inputs.permute(0,2,1)                        
        inputs, labels = inputs.to(device), labels.to(device)                        
        pred = model(inputs)                                            
        pred = F.softmax(pred, dim=1)
        predicted_labels = np.argmax(pred.detach().cpu().numpy(), axis=1)            
        labels = labels.cpu().numpy()            
        correct = np.sum(labels == predicted_labels)                            
        num = len(predicted_labels)                        
        total_num += num            
        total_correct += correct        

print(f"test accuracy after {epoch+1} epochs of training: total_correct={total_correct}, total_num={total_num} accuracy={total_correct/total_num}")

    
import matplotlib.pyplot as plt
plt.plot([l.item() for l in losses])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss.png")
torch.save(model.state_dict(), 'model_weights.pt')



    
    
    