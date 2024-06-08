import torch
import os.path
from colorama import init, Fore, Style
import pandas as pd
from Modules.logisticRegression import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class main():



    path = ""
    iteration = 1
    validationPercent = testPercent = 1
    learningRate = 0.1
    dataset = None
    #xTrain = xTest = yTrain = yTest = xVal = yVal = None
    batchSize = 20
    trainLoader = valLoader = testLoader = None
    scaler = MinMaxScaler()
    optimizerType = None
    device = None

    def colorText(self, text, color):
        init()
        colorCode = ""
        if color == "G":
            colorCode = "\033[32m"
        else:
            colorCode = "\033[31m"
        return f"{colorCode}{text}\033[0m"


    def checkGPU(self):
        global device
        if torch.cuda.is_available():
            print("CUDA is available")
            numberOfGpus = torch.cuda.device_count()
            print(f"Number of available GPUs: {numberOfGpus}")
            for i in range (numberOfGpus):
                gpuProperties = torch.cuda.get_device_properties(i)
                print(f"GPU{i}: {gpuProperties.name}, (CUDA cores: {gpuProperties.multi_processor_count})")
                device = torch.device("cuda")
            return True
        else:
            print("OOps! your GPU doesn't support required CUDA version.")
            return False
        
    def getDatasetPath(self):
        global path
        path = input("Where can i find the breast dataset?(Write the path to dataset):   ")
        if os.path.isfile(path + '/bdiag.csv'):
            print(self.colorText("Dataset exist", "G"))
        else:
            print(self.colorText("Dataset doesn't exist. Check the directory!", "R"))
    
    def getUserParams(self):
        global iteration, validationPercent, learningRate, testPercent, batchSize, optimizerType
        iteration = int(input("Enter iteration number: "))
        validationPercent = int(input("Enter validation percent: %"))/100
        testPercent = int(input("Enter test percent: %"))/100
        learningRate = float(input("Enter learning rate: "))
        batchSize = int(input("Enter batch size:(default 20): "))
        optimizerType = input("Which optimizer do you want to choose?(SGD/Adam): ")

    def loadDataFromCsv(self):
        #global dataset, xTrain, xTest, yTrain, yTest, xVal, yVal trainLoader = valLoader = testLoader = None
        global trainLoader, valLoader, testLoader
        dataset = pd.read_csv(path + '/bdiag.csv')
        print("A quick peek o dataset! ...\n")
        print(dataset.head())
        x = dataset.iloc[:, 2:32]
        y = dataset.iloc[:, 1]
        print(dataset.shape)
        print("Generating train, validation and test sets ...\n")
        xTrainTemp, xValTemp, yTrain, yVal = train_test_split(x, y, test_size=validationPercent)
        print(xTrainTemp.shape)
        print(yTrain.shape)
        xTrainTemp, xTestTemp, yTrain, yTest = train_test_split(xTrainTemp, yTrain, test_size=testPercent)
        print(xTrainTemp.shape)
        print(yTrain.shape)
        print(xValTemp.shape)
        print(yVal.shape)
        print(xTestTemp.shape)
        print(yTest.shape)
        print("Scaling data ...\n")
        xTrain = self.scaler.fit_transform(xTrainTemp)
        xVal = self.scaler.transform(xValTemp)
        xTest = self.scaler.transform(xTestTemp)
        print("Generationg Dataloader ...\n")
        labelEncoder = LabelEncoder()
        yTrainEncoded = labelEncoder.fit_transform(yTrain)
        yValEncoded = labelEncoder.fit_transform(yVal)
        yTestEncoded = labelEncoder.fit_transform(yTest)
        tensorXTrain = torch.Tensor(xTrain)
        tensorYTrain = torch.Tensor(yTrainEncoded).unsqueeze(1)
        tensorXVal = torch.Tensor(xVal)
        tensorYVal = torch.Tensor(yValEncoded).unsqueeze(1)
        tensorXTest = torch.Tensor(xTest)
        tensorYTest = torch.Tensor(yTestEncoded).unsqueeze(1)
        trainDataset = CustomDataset(tensorXTrain, tensorYTrain)
        valDataset = CustomDataset(tensorXVal, tensorYVal)
        testDataset = CustomDataset(tensorXTest, tensorYTest)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)
        testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    

    def startNN(self):
        if self.checkGPU():
            self.getDatasetPath()
            self.getUserParams()
            self.loadDataFromCsv()
            input_dim = 30
            output_dim = 1
            model = LogisticRegression(input_dim, output_dim).to(device)
            criterion = nn.BCELoss()
            if optimizerType == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
            training_losses = []
            print(f"Model is training on {iteration} of epochs")
            for epoch in range(iteration):
                for inputs, labels in trainLoader:
                    data, labels = data.to("cuda"), labels.to("cuda")
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                training_losses.append(loss.item())
            print(model)
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in valLoader:
                    data, labels = data.to("cuda"), labels.to("cuda")
                    outputs = model(inputs)
                    predicted = (outputs >= 0.5).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    accuracy = correct / total
                    print(f"Validation Accuracy: {accuracy:.2f}")
            model.train()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in testLoader:
                    data, labels = data.to("cuda"), labels.to("cuda")
                    outputs = model(inputs)
                    predicted = (outputs >= 0.5).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    accuracy = correct / total
                    print(f"Test Accuracy: {accuracy:.2f}")
            plt.plot(training_losses, label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            





if __name__ == '__main__':
    script = main()
    script.startNN()
