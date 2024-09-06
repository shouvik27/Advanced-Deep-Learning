import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from .model import CNN
from .dataloader import get_data_loaders

def train_pretext_task(num_epochs=5):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, _ = get_data_loaders()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for images, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

    return model

def fine_tune_model(model, num_epochs=5):
    model.fc2 = nn.Linear(128, 10)  # Adjust the last layer for digit classification

    for param in model.parameters():
        param.requires_grad = False
    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True

    optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loaders()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Fine-tuning Epoch [{epoch+1}/{num_epochs}]')
        for images, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

    return model

def evaluate_model(model):
    _, test_loader = get_data_loaders()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating')
        for images, labels in progress_bar:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
