import torch
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

# mnist
image_shape = [28, 28, 1]
data_path = '/data/mnist/'
train_x, train_y = torch.load(data_path + 'training.pt')
train_x, train_y = train_x.numpy(), train_y.numpy()
train_x = train_x[:, :, :, np.newaxis]
test_x, test_y = torch.load(data_path + 'test.pt')
test_x, test_y = test_x.numpy(), test_y.numpy()
test_x = test_x[:, :, :, np.newaxis]


def sample_valid(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x, y = x[idx], y[idx]

    x1, y1 = x[:-5000], y[:-5000]
    x2, y2 = x[-5000:], y[-5000:]

    return x1, y1, x2, y2

train_x, train_y, valid_x, valid_y = sample_valid(train_x, train_y)

train_x = train_x.reshape([-1, 28*28]).astype(np.float32) / 255.
valid_x = valid_x.reshape([-1, 28*28]).astype(np.float32) / 255.
test_x = test_x.reshape([-1, 28*28]).astype(np.float32) / 255.

model = GaussianMixture(n_components=100, random_state=0).fit(train_x)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)