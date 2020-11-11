import sys
from timeit import default_timer as timer

import torch
from torch.utils.data import Dataset
import numpy as np


class ImageCaptionDataset(Dataset):
  def __init__(self, pairs_dataset, cap_encodings):
    super(ImageCaptionDataset, self).__init__()
    self.pairs_dataset = pairs_dataset
    self.cap_encodings = cap_encodings

  def __len__(self):
    return len(self.pairs_dataset)

  def __getitem__(self, i):
    anchor, _ = self.pairs_dataset[i]
    # pos_enc = self.cap_encodings[i][random.randint(0, len(self.cap_encodings[i])-1)]
    pos_enc = self.cap_encodings[i][0]

    return anchor, pos_enc


def train_for_classification(net, train_loader, test_loader, optimizer, criterion, epochs=1, reports_every=1, device='cuda'):
  net.to(device)
  total_train = len(train_loader.dataset)
  total_test = len(test_loader.dataset)
  tiempo_epochs = 0
  train_loss, train_acc, test_acc = [], [], []

  for e in range(1,epochs+1):  
    inicio_epoch = timer()
    
    # Aseguramos que todos los parámetros se entrenarán usando .train()
    net.train()

    # Variables para las métricas
    running_loss, running_acc = 0.0, 0.0

    for i, data in enumerate(train_loader):
      # Desagregamos los datos y los pasamos a la GPU
      X, Y = data
      X, Y = X.to(device), Y.to(device)

      # Limpiamos los gradientes, pasamos el input por la red, calculamos
      # la loss, ejecutamos el backpropagation (.backward) 
      # y un paso del optimizador para modificar los parámetros
      optimizer.zero_grad()
      Y_logits = net(X)['fc_out']
      loss = criterion(Y_logits, Y)
      loss.backward()
      optimizer.step()

      # loss
      items = min(total_train, (i+1) * train_loader.batch_size)
      running_loss += loss.item()
      avg_loss = running_loss/(i+1)
      
      # accuracy
      _, max_idx = torch.max(Y_logits, dim=1)
      running_acc += torch.sum(max_idx == Y).item()
      avg_acc = running_acc/items*100

      # report
      sys.stdout.write(f'\rEpoch:{e}({items}/{total_train}), ' 
                       + f'Loss:{avg_loss:02.5f}, '
                       + f'Train Acc:{avg_acc:02.1f}%')

    if e % reports_every == 0:
      sys.stdout.write(', Validating...')
      train_loss.append(avg_loss)
      train_acc.append(avg_acc)
      net.eval()
      running_acc = 0.0
      for i, data in enumerate(test_loader):
        X, Y = data
        X, Y = X.to(device), Y.to(device)
        Y_logits = net(X)['fc_out']
        _, max_idx = torch.max(Y_logits, dim=1)
        running_acc += torch.sum(max_idx == Y).item()
        avg_acc = running_acc/total_test*100
      test_acc.append(avg_acc)
      sys.stdout.write(f', Val Acc:{avg_acc:02.2f}%.\n')
    else:
      sys.stdout.write('\n')

  return train_loss, train_acc, test_acc

  
def l2norm(x):
  norm = np.linalg.norm(x, axis=1, keepdims=True)
  return 1.0 * x / norm


def compute_ranks_x2y(x, y):
  dists = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)
  ranks = torch.zeros(dists.shape[0])
  for i in range(len(ranks)):
    d_i = dists[i,:]
    inds = torch.argsort(d_i)
    rank = torch.where(inds == i)[0][0]
    ranks[i] = rank
  return ranks


def train_for_retrieval(img_net, text_net, train_loader, test_loader, optimizer, criterion, epochs=1, reports_every=1, device='cuda', norm=True):
  img_net.to(device)
  text_net.to(device)

  total_train = len(train_loader.dataset)
  total_test = len(test_loader.dataset)
  tiempo_epochs = 0
  train_loss, train_meanrr, test_meanrr = [], [], []

  for e in range(1,epochs+1):
    inicio_epoch = timer()

    # Aseguramos que todos los parámetros se entrenarán usando .train()
    img_net.train()
    text_net.train()

    # Variables para las métricas
    running_loss, running_meanrr = 0.0, 0.0

    for i, data in enumerate(train_loader):
      # Desagregamos los datos y los pasamos a la GPU
      a, p = data
      if norm:
        a, p = l2norm(a), l2norm(p)
      a, p = a.to(device), p.to(device)

      # Limpiamos los gradientes, pasamos el input por la red, calculamos
      # la loss, ejecutamos el backpropagation (.backward) 
      # y un paso del optimizador para modificar los parámetros
      optimizer.zero_grad()

      a_enc = img_net(a)['fc_out']
      p_enc = text_net(p)['fc_out']

      loss = criterion(a_enc, p_enc)
      loss.backward()
      optimizer.step()

      # loss
      items = min(total_train, (i+1) * train_loader.batch_size)
      running_loss += loss.item()
      avg_loss = running_loss/(i+1)

      # mean-rank
      ranks = compute_ranks_x2y(a_enc, p_enc)
      # running_meanr += (ranks.mean()/len(a))
      running_meanrr += (torch.reciprocal(ranks+1).mean())
      avg_meanrr = running_meanrr/(i+1)

      # report
      sys.stdout.write(f'\rEpoch:{e}({items}/{total_train}), ' 
                       + f'Loss:{avg_loss:02.5f}, '
                       + f'Train MRR:{avg_meanrr:02.3f}')

    if e % reports_every == 0:
      sys.stdout.write(', Validating...')
      train_loss.append(avg_loss)
      train_meanrr.append(avg_meanrr)

      img_net.eval()
      text_net.eval()

      running_meanrr = 0.0
      for i, data in enumerate(test_loader):
        a, p = data
        a, p = a.to(device), p.to(device)

        a_enc = img_net(a)['fc_out']
        p_enc = text_net(p)['fc_out']

        # mean-rank
        ranks = compute_ranks_x2y(a_enc, p_enc)
        # running_meanrr += (ranks.mean()/len(a))
        running_meanrr += (torch.reciprocal(ranks+1).mean())
        avg_meanrr = running_meanrr/(i+1)
      test_meanrr.append(avg_meanrr)
      sys.stdout.write(f', Val MRR:{avg_meanrr:02.3f}.\n')
    else:
      sys.stdout.write('\n')

  return train_loss, train_meanrr, test_meanrr
