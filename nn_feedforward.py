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

#%%
def load_files_thetas(integer):
    folder_path = "simulation_results_fixed_design_no_chirp_new"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            if filename.startswith("params"):
                theta = np.load(os.path.join(folder_path, filename))
    return theta[:9]

def load_files(integer):
    folder_path = "summary_statistics_fixed_design_no_chirp_new"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            y = np.load(os.path.join(folder_path, filename))
    return y  

#%% load in data
n_samples = 10000
n_theta = 9
n_y = 105
x = np.zeros((n_samples,n_theta))
#y = np.zeros((n_samples,n_y))
for i in range(n_samples):
    x[i,:] = load_files_thetas(i)
    #y[i,:] = load_files(i)
    
#y = y[:,1:]

#%% load in data
y_all = np.load("summary_statistics_fixed_design_no_chirp_new/y.npy")
x = np.load("summary_statistics_fixed_design_no_chirp_new/theta9.npy")
n_theta = 9
y = np.delete(y_all,np.where(np.var(y_all,axis=0)<1e-12),axis=1)
n_y = y.shape[1]

#%% scale and test train split
from sklearn.preprocessing import StandardScaler
x_scalers = []
y_scalers = []
x_scaled  = np.zeros(x.shape)
y_scaled  = np.zeros(y.shape)
for i in range(n_theta):
    x_scaler = StandardScaler()
    x_scaler.fit(x[:,i].reshape(-1,1))
    x_scalers.append(x_scaler)
    x_scaled[:,i]  = x_scaler.transform(x[:,i].reshape(-1,1)).flatten()
   
for j in range(n_y):#range(n_y-1):
    y_scaler = StandardScaler()
    y_scaler.fit(y[:,j].reshape(-1,1))
    y_scalers.append(y_scaler) 
    y_scaled[:,j]  = y_scaler.transform(y[:,j].reshape(-1,1)).flatten()

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
X_train, y_train, X_test, y_test = map(torch.tensor, (X_train.astype(np.float32), y_train.astype(np.float32), X_test.astype(np.float32), y_test.astype(np.float32)))

#%% define model

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
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(42)
    self.net = nn.Sequential(
        nn.Linear(9, 64), 
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 98*2),
        nn.ReLU(),
        nn.Linear(98*2, 104)
    )

  def forward(self, X):
    return self.net(X)

  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred

model = feed_forward().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#%% train network
X_train=X_train.to(device)
y_train=y_train.to(device)
X_test=X_test.to(device)
y_test=y_test.to(device)

def fit_v2(x, y, model, opt, loss_fn, epochs = 1500):
  
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
plt.hlines(r_squareds_real.mean(),-5,105, color='gray',label='Average')
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
index = 11
plt.figure()
plt.title('NN Model Prediction Accuracy, $R^{2} = $'+str(r_squareds[index].item())[:6])
plt.xlabel("Test Data (scaled)")
plt.ylabel("Model Predictions (scaled)")
plt.scatter(actual.cpu().detach().numpy()[:,index],pred.cpu().detach().numpy()[:,index], s=5)

#%% chirp signal

def load_files_thetas_chirp(integer):
    folder_path = "simulation_results_fixed_design_new"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            if filename.startswith("params"):
                theta = np.load(os.path.join(folder_path, filename))
    return theta[:9]

def load_files_chirp(integer):
    folder_path = "summary_statistics_fixed_design_new"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            y = np.load(os.path.join(folder_path, filename))
    return y  

#%% load in data
n_samples = 10000
n_theta = 9
n_y = 225
x = np.zeros((n_samples,n_theta))
#y = np.zeros((n_samples,n_y))
for i in range(n_samples):
    x[i,:] = load_files_thetas_chirp(i)
    #y[i,:] = load_files_chirp(i)

#%% load in data
y = np.load("summary_statistics_fixed_design_new/y_reduced.npy")
x = np.load("summary_statistics_fixed_design_new/theta_9total.npy")
n_theta = 9
n_y = y.shape[1]

#%% scale and test train split
from sklearn.preprocessing import StandardScaler
x_scalers = []
y_scalers = []
x_scaled  = np.zeros(x.shape)
y_scaled  = np.zeros(y.shape)
for i in range(n_theta):
    x_scaler = StandardScaler()
    x_scaler.fit(x[:,i].reshape(-1,1))
    x_scalers.append(x_scaler)
    x_scaled[:,i]  = x_scaler.transform(x[:,i].reshape(-1,1)).flatten()
   
for j in range(n_y):#range(n_y-1):
    y_scaler = StandardScaler()
    y_scaler.fit(y[:,j].reshape(-1,1))
    y_scalers.append(y_scaler) 
    y_scaled[:,j]  = y_scaler.transform(y[:,j].reshape(-1,1)).flatten()

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
X_train, y_train, X_test, y_test = map(torch.tensor, (X_train.astype(np.float32), y_train.astype(np.float32), X_test.astype(np.float32), y_test.astype(np.float32)))

#%% define model

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
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(42)
    self.net = nn.Sequential(
        nn.Linear(9, 64), 
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 184*2),
        nn.ReLU(),
        nn.Linear(184*2, 184)
    )

  def forward(self, X):
    return self.net(X)

  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred

model = feed_forward().to(device)

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
    
#%% compiling vector of losses for each output
losses = torch.zeros([n_y], dtype=torch.float32)
r_squareds = torch.zeros([n_y], dtype=torch.float32)
for output in range(n_y):
    losses[output] = loss_fn(pred[:,output],actual[:,output])
    r_squareds[output] = torch.corrcoef(torch.stack((pred[:,output],actual[:,output])))[0,1]
    
#%% repeated on training data
model.eval()
with torch.no_grad():
    X_train = X_train.to(device)
    pred_train = model(X_train)
    actual_train = y_train
    
#%% compiling vector of losses for each output
losses_train = torch.zeros([n_y], dtype=torch.float32)
r_squareds_train = torch.zeros([n_y], dtype=torch.float32)
for output in range(n_y):
    losses_train[output] = loss_fn(pred_train[:,output],actual_train[:,output])
    r_squareds_train[output] = torch.corrcoef(torch.stack((pred_train[:,output],actual_train[:,output])))[0,1]


#%%
plt.figure()
plt.plot(losses,'*',label='Test Losses')
plt.plot(losses_train, '.', label = 'Training Losses')
plt.xlabel('Summary Statistic Index')
plt.ylabel('MSE Loss')
plt.legend()

#%%
plt.figure()
plt.plot(r_squareds,'*',label='Test Losses')
plt.plot(r_squareds_train, '.', label = 'Training Losses')
plt.xlabel('Summary Statistic Index')
plt.ylabel(r'$R^{2}$')
plt.legend()
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