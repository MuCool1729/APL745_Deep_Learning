from sklearn.model_selection import learning_curve
import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import time
# A Neural Net with 1-hidden layer network with 50 hidden units

class NNet1(nn.Module):
    def __init__(self):
            super(NNet1, self).__init__()
            self.input_size = 28*28
            self.L1 = nn.Linear(self.input_size, 50) 
            self.activation = nn.Sigmoid()
            self.L2 = nn.Linear(50, 10)  

    def forward(self, x):
            x = self.L1(x)
            x = self.activation(x)
            out = self.L2(x)
            return out
class Model:
    def __init__(self,lr=0.001,wd=0.005,epoch=100):
        self.model = NNet1()
        self.learning_rate = lr
        self.weight_decay = wd
        self.num_of_epochs = epoch
    def train(self,train_dataloader):
        self.lossfun = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.learning_rate,weight_decay=self.weight_decay)
        for epoch in range(self.num_of_epochs):
            for i, (inputs,labels) in enumerate(train_dataloader):
                outputs = self.model(inputs)
                loss = self.lossfun(outputs,labels)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'epoch: {epoch+1}/{self.num_of_epochs},loss = {loss.item():.4f}')
    
    def test(self,test_dataloader):
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs,labels in test_dataloader:
                outputs = self.model(inputs)
                max_val,predicted_class = torch.max(outputs,1)
                total += labels.shape[0]
                correct += (predicted_class==labels).sum().item()
            accuracy = 100.0 * (correct / total)
            print(f'accuracy={accuracy}')
    def print_model_congiurations(self):
        summary(self.model,(784,))
    def print_first_five_preds(self,test_dataloader):
        with torch.no_grad():
            for input,labels in test_dataloader:
                out = self.model(input)
                _,pred_class = torch.max(out,1)
                for i in range(5):
                    image = input[i]
                    label = "ground truth is: " + str(labels[i].item())+ " and predicted label is: " + str(pred_class[i].item())
                    image = np.reshape(image,(28,28))
                    plt.imshow(image, interpolation='nearest',cmap = 'gray')
                    plt.xlabel(label,color='yellow')
                    plt.show()
                break
    
    def compute(self,train_dataloader,test_dataloader):
        self.train_losses = []
        self.test_accuracies = []
        self.train_accuracies = []
        self.test_losses = []
        start = time.time()
        self.lossfun = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate,weight_decay=self.weight_decay)
        for epoch in range(self.num_of_epochs):
            losses = []
            train_correct = 0;
            train_total = 0;
            for i, (inputs,labels) in enumerate(train_dataloader):
                outputs = self.model(inputs)
                _,pred_class = torch.max(outputs,1)
                train_total += labels.shape[0]
                train_correct += (pred_class==labels).sum().item()
                loss = self.lossfun(outputs,labels)
                self.optimizer.zero_grad()
                #self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            self.train_accuracies.append(100.0*train_correct/train_total)
            with torch.no_grad():
                correct = 0
                total = 0
                lossess = []
                for inputs,labels in test_dataloader:
                    outputs = self.model(inputs)
                    lossess.append(self.lossfun(outputs,labels).item());
                    max_val,predicted_class = torch.max(outputs,1)
                    total += labels.shape[0]
                    correct += (predicted_class==labels).sum().item()
                self.test_accuracies.append( 100.0 * (correct / total))
                self.test_losses.append(sum(lossess))   
            self.train_losses.append(sum(losses))
            print(f'epoch: {epoch+1}/{self.num_of_epochs},train_loss = {sum(losses):.4f}, test_loss = {self.test_losses[-1]:.4f} , train_acc = {self.train_accuracies[-1]:.2f},test_acc = {self.test_accuracies[-1]:.2f}')
        self.time_taken = time.time()-start;
    def draw_graphs(self):
        plt.plot([(i+1) for i in range(self.num_of_epochs)],self.train_losses , label ="train")        
        plt.plot([(i+1) for i in range(self.num_of_epochs)],self.test_losses , label = "test")
        plt.title('training and test losses v/s epochs')
        plt.xlabel('epochs')
        plt.ylabel('losses') 
        plt.legend()       
        plt.show()
        plt.plot([(i+1) for i in range(self.num_of_epochs)],self.train_accuracies, label="train_acc")        
        plt.plot([(i+1) for i in range(self.num_of_epochs)],self.test_accuracies, label="test_acc")        
        plt.title('training and test accuracies v/s epochs')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()       
        plt.show()
    def time_taken(self):
        return self.time_taken;    