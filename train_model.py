import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from sklearn.model_selection import train_test_split

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_layers=3, num_classes=3):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class SignLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(data_dir='data'):
    sequences = []
    labels = []
    actions = []
    
    print(f"\nReading data from: {data_dir}")
    
    for action in sorted(os.listdir(data_dir)):
        action_path = os.path.join(data_dir, action)
        if not os.path.isdir(action_path):
            continue
        
        print(f"   {action}...", end=' ')
        actions.append(action)
        action_idx = len(actions) - 1
        
        sequence_count = 0
        
        for sequence_folder in os.listdir(action_path):
            sequence_path = os.path.join(action_path, sequence_folder)
            
            if not os.path.isdir(sequence_path):
                continue
            
            frames = []
            
            frame_files = sorted(
                [f for f in os.listdir(sequence_path) if f.endswith(".npy")],
                key=lambda x: int(x.split(".")[0])
            )
            
            for file in frame_files:
                file_path = os.path.join(sequence_path, file)
                frame = np.load(file_path)
                frames.append(frame)
            
            if len(frames) > 0:
                sequences.append(np.array(frames))
                labels.append(action_idx)
                sequence_count += 1
        
        if sequence_count == 0:
            print("Empty!")
        else:
            print(f"{sequence_count} sequences")
    
    if len(sequences) == 0:
        raise Exception("No data found!")
    
    return np.array(sequences), np.array(labels), np.array(actions)

def train_model(epochs=100, batch_size=32, learning_rate=0.001, data_dir='data'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    try:
        X, y, actions = load_data(data_dir)
        print(f"\nLoaded {len(X)} samples")
        print(f"Classes: {list(actions)}")
    except Exception as e:
        print(f"\n{e}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"   - Training: {len(X_train)} samples")
    print(f"   - Testing: {len(X_test)} samples")
    
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train.shape[2]
    num_classes = len(actions)
    
    model = SignLanguageLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=3,
        num_classes=num_classes
    ).to(device)
    
    print(f"\nModel:")
    print(f"   - Input: {input_size}")
    print(f"   - Hidden: 64")
    print(f"   - Layers: 3")
    print(f"   - Classes: {num_classes}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n{'='*60}")
    print(f"Starting training ({epochs} Epochs)")
    print(f"{'='*60}\n")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {train_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': accuracy,
                'input_size': input_size,
                'hidden_size': 64,
                'num_layers': 3,
                'num_classes': num_classes
            }, 'sign_language_model.pth')
            
            with open('actions.pkl', 'wb') as f:
                pickle.dump(actions, f)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Saved files:")
    print(f"   - sign_language_model.pth")
    print(f"   - actions.pkl")
    print(f"\nNext step: python sign_detection.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sign Language Recognition Model Training")
    print("="*60)
    
    try:
        epochs = int(input("\nNumber of Epochs (default 100): ") or "100")
        batch_size = int(input("Batch size (default 32): ") or "32")
        learning_rate = float(input("Learning rate (default 0.001): ") or "0.001")
    except:
        epochs = 100
        batch_size = 32
        learning_rate = 0.001
        print("\nUsing default settings")
    
    train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)


