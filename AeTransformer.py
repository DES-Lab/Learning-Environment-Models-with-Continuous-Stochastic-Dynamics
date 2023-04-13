import random

import torch
from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class AeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, input_space, latent_space, epochs):
        super().__init__()
        self.input_space = input_space
        self.latent_space = latent_space
        self.ae = Autoenc(input_space,latent_space)
        self.epochs = epochs
        self.power_transformer = PowerTransformer()
        self.use_power_transform = True

    def fit(self, X, y=None):
        model = self.ae
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
        outputs = []
        losses = []
        sample = []
        from guppy import hpy
        h = hpy()
        for i in range(X.shape[0]):
            input_elem = torch.from_numpy(X[i, :])
            sample.append(input_elem)
        sample = random.choices(sample,k=100000)
        # sample_points = random.choices(range(X.shape[0]),k=10000)
        # random.shuffle(sample)
        for epoch in range(self.epochs):
            # print(torch.cuda.memory_allocated())
            print(model.encoder[0].weight)
            for input_elem in sample:
                # input_elem = torch.from_numpy(X[point,:])
                reconstructed = model(input_elem)
                loss = loss_function(reconstructed, input_elem)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch}: {avg_loss}")
            outputs.append((epoch, avg_loss))
        for (e,l) in outputs:
            print(f"Epoch {e}: {l}")
        if self.use_power_transform:
            self.power_transformer.fit(self.ae(torch.from_numpy(X)).detach().numpy())
    def transform(self, X, y=None):
        # Perform arbitary transformation
        # transformed = np.zeros((x.shape[0], 4))
        if self.use_power_transform:
            return self.power_transformer.transform(self.ae(torch.from_numpy(X)).detach().numpy())
        else:
            return self.ae(torch.from_numpy(X)).detach().numpy()


class Autoenc(torch.nn.Module):
   def __init__(self, input_space, latent_space):
      super().__init__()
      self.input_space = input_space
      self.latent_space = latent_space

      self.encoder = torch.nn.Sequential(
          torch.nn.Linear(input_space, 6),
          torch.nn.ReLU(),
          torch.nn.Linear(6, self.latent_space)
      # torch.nn.Linear(6, self.latent_space)
      ).double()

      self.decoder = torch.nn.Sequential(
      torch.nn.Linear(self.latent_space, 6),
          torch.nn.ReLU(),
         torch.nn.Linear(6, input_space),
          # torch.nn.Sigmoid()
         # torch.nn.ReLU(),
         #  torch.nn.Linear(6, input_space),
         #  torch.nn.Sigmoid()
      ).double()
   def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded


