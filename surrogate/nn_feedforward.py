"""
`Learn the Basics <intro.html>`_ ||
**Quickstart** ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Quickstart
===================
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

Working with data
-----------------
PyTorch has two `primitives to work with data <https://pytorch.org/docs/stable/data.html>`_:
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.

"""
#%%
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

#%% read in files, convert to torch tensor
df_in = pd.read_csv("parameters_and_design_sep24_5272_runs.csv")
df_out = pd.read_csv("summary_statistics_sep24_5272_runs.csv")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu') 

X = torch.from_numpy(df_in.values).double().to(device)#[:2000,:]
y = torch.from_numpy(df_out.values).double().to(device)#[:2000,:2]
n_theta = X.shape[1]
n_y = y.shape[1]

#%% cleaning y data
pos_inds = torch.tensor([0, 1, 3, 5, 7, 9, 10, 12, 13, 16, 18, 19, 22, 24, 25, 28, 30])
n_repeats = 31 # summary stats repeat 31 times
y_cleaned = y
for j in range(n_y):#range(n_y-1):
    # cleaning y data for large values, set them to 2*stdev from median
    thresh = 1e5
    quantile = y_cleaned[:,j].median() + 2*y_cleaned[np.where(y_cleaned[:,j].abs()<thresh),j].std()
    y_cleaned[np.where(y_cleaned[:,j].abs()>thresh),j] = quantile
    
    # apply logarithm to strictly positive indices
    if (pos_inds == np.remainder(j,31)).sum():
        y_cleaned[:,j] = y_cleaned[:,j].log()

#%% scale and test train split
from sklearn.preprocessing import StandardScaler
x_scalers = []
y_scalers = []
x_scaled  = np.zeros(X.shape)
y_scaled  = np.zeros(y_cleaned.shape)
for i in range(n_theta):
    x_scaler = StandardScaler()
    x_scaler.fit(X[:,i].reshape(-1,1))
    x_scalers.append(x_scaler)
    x_scaled[:,i]  = x_scaler.transform(X[:,i].reshape(-1,1)).flatten()
   
for j in range(n_y):#range(n_y-1):
    y_scaler = StandardScaler()
    y_scaler.fit(y_cleaned[:,j].reshape(-1,1))
    y_scalers.append(y_scaler) 
    y_scaled[:,j]  = y_scaler.transform(y_cleaned[:,j].reshape(-1,1)).flatten()

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
X_train, y_train, X_test, y_test = map(torch.tensor, (X_train.astype(np.float32), y_train.astype(np.float32), X_test.astype(np.float32), y_test.astype(np.float32)))

#%% define model
import torch.nn.functional as F
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(8, 16)
#         # Inputs to hidden layer linear transformation
#         self.hidden2 = nn.Linear(16, 16)
#         # Output layer, 10 units - one for each digit
#         self.output = nn.Linear(16, 1)
        
#     def forward(self, x):
#         # Hidden layer with sigmoid activation
#         x = nn.functional.sigmoid(self.hidden(x))
#         x = nn.functional.sigmoid(self.hidden2(x))
#         # Output layer with softmax activation
#         x = nn.functional.sigmoid(self.output(x))
        
#         return x
    
#     def predict(self, X):
#       Y_pred = self.forward(X)
#       return Y_pred

class feed_forward(nn.Module): 
  def __init__(self, n_x, n_hidden, n_y, seed=42):
    super().__init__()
    torch.manual_seed(seed)
    self.l1 = nn.Linear(n_x, n_hidden)
    self.l2 = nn.Linear(n_hidden, n_hidden)
    self.l3 = nn.Linear(n_hidden, n_hidden)
    self.l4 = nn.Linear(n_hidden, n_hidden)
    self.l5 = nn.Linear(n_hidden, n_y)
    # self.net = nn.Sequential(
    #     nn.Linear(n_x, n_hidden), 
    #     nn.ReLU(),
    #     nn.Linear(n_hidden, n_hidden),
    #     nn.ReLU(),
    #     nn.Linear(n_hidden, n_y)
    # )
  # def forward(self, X):
  #   return self.net(X)
  def forward(self, X):
      out = F.relu(self.l1(X))
      out2 = F.relu(self.l2(out))
      out3 = F.relu(self.l3(out2))
      out4 = F.relu(self.l4(out3))
      out5 = self.l5(out4)
      return out5
  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred
    
class feed_forward_dropout(nn.Module):  
  def __init__(self, n_x, n_hidden, n_y, seed=42, dropout_rate=0.):
    super().__init__()
    torch.manual_seed(seed)
    self.l1 = nn.Linear(n_x, n_hidden)
    self.bn1 = nn.BatchNorm1d(n_hidden)
    self.l2 = nn.Linear(n_hidden, n_hidden)
    self.l3 = nn.Linear(n_hidden, n_hidden)
    self.l4 = nn.Linear(n_hidden, n_hidden)
    self.l5 = nn.Linear(n_hidden, n_y)
    self.dropout = nn.Dropout(dropout_rate) 
  def forward(self, X):
      out = F.relu(self.bn1(self.l1(X)))
      out2 = F.relu(self.l2(out))
      out3 = F.relu(self.l3(out2))
      out4 = F.relu(self.l4(out3))
      out5 = self.l5(out4)
      return out5
  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred

class feed_forward_bn(nn.Module): 
  def __init__(self, n_x, n_hidden, n_y, seed=42, dropout_rate=0.):
    super().__init__()
    torch.manual_seed(seed)
    self.l1 = nn.Linear(n_x, n_hidden)
    self.bn1 = nn.BatchNorm1d(n_hidden)
    self.l2 = nn.Linear(n_hidden, n_hidden)
    self.bn2 = nn.BatchNorm1d(n_hidden)
    self.l3 = nn.Linear(n_hidden, n_y)
    self.bn3 = nn.BatchNorm1d(n_y)
    self.dropout = nn.Dropout(dropout_rate) 
  def forward(self, X):
      out = F.relu(self.bn1(self.l1(X)))
      out2 = F.relu(self.bn2(self.l2(out)))
      out3 = self.dropout(self.bn3(self.l3(out2)))
      return out3
  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred

#model = feed_forward(n_theta, n_theta*8, n_y).to(device)
model = feed_forward_bn(n_theta, n_theta*8, n_y, seed=43, dropout_rate=0.5).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#%% train network
X_train=X_train.to(device)
y_train=y_train.to(device)
X_test=X_test.to(device)
y_test=y_test.to(device)

def fit_v2(x, y, model, opt, loss_fn, epochs = 1000):
  
  for epoch in range(epochs):
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if epoch%100==0:
        print(epoch)
        print(loss)
    
  return loss.item()

print('Final loss', fit_v2(X_train, y_train, model, optimizer, loss_fn))

#%% testing
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    pred = model(X_test)
    predicted = pred
    actual = y_test

#%% sanity check plot, should be clustered around y=x 
index = 1
plt.figure()
plt.title('NN Model Prediction Accuracy')
plt.xlabel("Test Data (scaled)")
plt.ylabel("Model Predictions (scaled)")
plt.scatter(actual.cpu().detach().numpy()[:,index],pred.cpu().detach().numpy()[:,index], s=5)

#%%
# losses = torch.zeros([n_y], dtype=torch.float32)
# r_squareds = torch.zeros([n_y], dtype=torch.float32)
# for output in range(n_y):
#     r_squareds[output] = np.corrcoef(np.vstack((pred[:,output].cpu().detach().numpy(),actual[:,output].cpu().detach().numpy())))[0,1]
#     losses[output] = torch.from_numpy(np.array([((pred[:,output].cpu().detach().numpy() - actual[:,output].cpu().detach().numpy())**2).mean(axis=0)]))

#%% convert to real units
actual_real = np.zeros((actual.shape[0],n_y))
pred_real = np.zeros((pred.shape[0],n_y))
for j in range(n_y):
    actual_np = actual[:,j].cpu().detach().numpy().reshape(-1,1)
    actual_real[:,j] = y_scalers[j].inverse_transform(actual_np).flatten()
    pred_np = pred[:,j].cpu().detach().numpy().reshape(-1,1)
    pred_real[:,j] = y_scalers[j].inverse_transform(pred_np).flatten()
#%%
losses_real = torch.zeros([n_y], dtype=torch.float32)
r_squareds_real = torch.zeros([n_y], dtype=torch.float32)
for output in range(n_y):
    r_squareds_real[output] = np.corrcoef(np.vstack((pred_real[:,output],actual_real[:,output])))[0,1]
    losses_real[output] = ((pred_real[:,output] - actual_real[:,output])**2).mean(axis=0)
    
plt.figure()
plt.title(r'$R^{2}$ for each NN Output, in real units')
plt.xlabel('Output Index')
plt.ylabel(r'$R^{2}$' )
plt.plot(r_squareds_real, 'o',markersize=5)
plt.hlines(r_squareds_real.mean(),-5,n_y+5, color='gray',label='Average')
plt.legend()
#%%
plt.figure()
plt.title("MSE Loss for each NN Output, in real units")
plt.xlabel('Output Index')
plt.ylabel('MSE')
plt.plot(losses_real, '.')
#%% compiling vector of losses for each output
losses = torch.zeros([n_y], dtype=torch.float32)
r_squareds = torch.zeros([n_y], dtype=torch.float32)
for output in range(n_y):
    losses[output] = loss_fn(pred[:,output],actual[:,output])
    r_squareds[output] = torch.corrcoef(torch.stack((pred[:,output],actual[:,output])))[0,1]
    
#%% some plots
plt.rcParams['figure.dpi'] = 600
plt.style.use('bmh')

plt.figure()
plt.title("MSE Loss for each NN Output")
plt.xlabel('Output Index')
plt.ylabel('MSE')
plt.plot(losses, 'o')
#%%
plt.figure()
plt.title(r'$R^{2}$ for each NN Output')
plt.xlabel('Output Index')
plt.ylabel(r'$R^{2}$' )
plt.plot(r_squareds, 'o',markersize=5)
plt.hlines(r_squareds.mean(),-5,102, color='gray',label='Average = 0.986')
plt.legend()

#%% 
index = 27
plt.figure()
plt.title('NN Model Prediction Accuracy, $R^{2} = $'+str(r_squareds[index].item())[:6])
plt.xlabel("Test Data (scaled)")
plt.ylabel("Model Predictions (scaled)")
plt.scatter(actual.cpu().detach().numpy()[:,index],pred.cpu().detach().numpy()[:,index], s=5)

#%%
######################################################################
# We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
# automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
# in the dataloader iterable will return a batch of 64 features and labels.

# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# ######################################################################
# # Read more about `loading data in PyTorch <data_tutorial.html>`_.
# #

# ######################################################################
# # --------------
# #

# ################################
# # Creating Models
# # ------------------
# # To define a neural network in PyTorch, we create a class that inherits
# # from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. We define the layers of the network
# # in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
# # operations in the neural network, we move it to the GPU or MPS if available.

# # Get cpu, gpu or mps device for training.
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork().to(device)
# print(model)

# ######################################################################
# # Read more about `building neural networks in PyTorch <buildmodel_tutorial.html>`_.
# #


# ######################################################################
# # --------------
# #


# #####################################################################
# # Optimizing the Model Parameters
# # ----------------------------------------
# # To train a model, we need a `loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# # and an `optimizer <https://pytorch.org/docs/stable/optim.html>`_.

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# #######################################################################
# # In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
# # backpropagates the prediction error to adjust the model's parameters.

# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# ##############################################################################
# # We also check the model's performance against the test dataset to ensure it is learning.

# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# ##############################################################################
# # The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
# # parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
# # accuracy increase and the loss decrease with every epoch.

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# ######################################################################
# # Read more about `Training your model <optimization_tutorial.html>`_.
# #

# ######################################################################
# # --------------
# #

# ######################################################################
# # Saving Models
# # -------------
# # A common way to save a model is to serialize the internal state dictionary (containing the model parameters).

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")



# ######################################################################
# # Loading Models
# # ----------------------------
# #
# # The process for loading a model includes re-creating the model structure and loading
# # the state dictionary into it.

# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))

# #############################################################
# # This model can now be used to make predictions.

# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')


######################################################################
# Read more about `Saving & Loading your model <saveloadrun_tutorial.html>`_.
#