#transformers
train_transform = transforms.Compose([
    transforms.Lambda(lambda img:img.convert("L")),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=7),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor()])
test_transform = transforms.Compose([
    transforms.Lambda(lambda img:img.convert("L")),
    transforms.Resize(size =(224,224)),
    transforms.ToTensor()])

#making datasets
from torchvision import datasets
train_data = datasets.ImageFolder(
    root = train_dir,
    transform = train_transform,
    target_transform = None)
test_data = datasets.ImageFolder(
    root = test_dir,
    transform = test_transform,
    target_transform = None)
print(f"Train Data Length:{len(train_data)} | Test Data Length:{len(test_data)}")

#dataloader
from torch.utils.data import DataLoader
import os
BATCH_SIZE = 16
NUM_WORKERS = 2
g = torch.Generator().manual_seed(88)
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              num_workers = NUM_WORKERS,
                              shuffle = True,
                              generator = g)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size = BATCH_SIZE,
                             num_workers = NUM_WORKERS)
print(f"Length Of Train Dataloader:{len(train_dataloader)} | Length Of Test Dataloader:{len(test_dataloader)}")

#Architecture
class pneumonia(nn.Module):
  def __init__(self,input_shape,output_shape):
    super().__init__()
    self.convblock1 = nn.Sequential(
    nn.Conv2d(in_channels = input_shape,
              out_channels = 16,
              kernel_size = 3,
              stride = 1,
              padding = 1,),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(in_channels = 16,
              out_channels = 32,
              kernel_size = 3,
              stride = 1,
              padding =1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2,stride = 2))
    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels = 32,
                  out_channels = 32,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels = 32,
                  out_channels = 64,
                  kernel_size =3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2,stride = 2))
    self.convblock3 = nn.Sequential(
        nn.Conv2d(in_channels = 64,
                  out_channels = 64,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64,
                  out_channels = 128,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2,stride = 2))
    self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels = 128,
                  out_channels = 128,
                  kernel_size  =3,
                  stride = 1,
                  padding =1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels = 128,
                  out_channels = 256,
                  kernel_size= 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2,stride = 2))
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Dropout(p=0.4),
        nn.Linear(in_features = 256,
                  out_features = 128),
        nn.ReLU(),
        nn.Linear(in_features = 128,
                  out_features = 64),
        nn.ReLU(),
        nn.Linear(in_features = 64,
                  out_features = 32),
        nn.ReLU(),
        nn.Linear(in_features = 32,
                  out_features = output_shape))
  def forward(self,x):
    return self.classifier(self.convblock4(self.convblock3(self.convblock2(self.convblock1(x)))))

#train step
def train_step(model,dataloader,loss_fn,optimizer,device = device):
  model.train()
  train_loss,train_acc = 0,0
  for Batch,(X,y) in enumerate(dataloader):
    X,y = X.to(device),y.to(device)
    y_predlogits = model(X)
    loss = loss_fn(y_predlogits,y)
    train_loss+=loss.item() * len(y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_pred_class = y_predlogits.argmax(dim = 1)
    train_acc += (y_pred_class == y).sum().item()
  train_loss = train_loss/len(dataloader.dataset)
  train_acc = train_acc/len(dataloader.dataset)
  return train_loss,train_acc

#test step
def test_step(model, dataloader, loss_fn, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for Batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item() * len(y)
            y_pred_class = y_pred_logits.argmax(dim=1)
            test_acc += (y_pred_class == y).sum().item()
    test_loss = test_loss / len(dataloader.dataset)
    test_acc = test_acc / len(dataloader.dataset)
    return test_loss, test_acc

#train step
def train(model,test_dataloader,optimizer,train_dataloader,loss_fn,epochs):
    results = {
        "train_loss":[],
        "train_acc":[],
        "test_loss":[],
        "test_acc":[]}
    best_model_path = "best_pneumonia_model.pth"
    best_test_acc = 0
    for epoch in range(epochs):
        train_loss,train_acc = train_step(model=model,
                                        dataloader = train_dataloader,
                                        loss_fn = loss_fn,
                                        optimizer = optimizer)
        test_loss,test_acc = test_step(model=model,
                                       dataloader = test_dataloader,
                                       loss_fn=loss_fn)
        print(f"Epoch:{epoch+1} | trainloss:{train_loss:.4f} | train_acc:{train_acc:.4f} | test_loss:{test_loss:.4f} | test_acc:{test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if test_acc > best_test_acc:
          best_test_acc = test_acc
          torch.save(model.state_dict(), best_model_path)
          print(f"   → Saved best model: {test_acc:.4f} at epoch {epoch+1}")
    return results

#training
NUM_EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                                lr =0.0001,
                             weight_decay=1e-4)
model_results = train(
    model = model,
    test_dataloader=test_dataloader,
    optimizer = optimizer,
    train_dataloader=train_dataloader,
    loss_fn=loss_fn,
    epochs = NUM_EPOCHS
)
