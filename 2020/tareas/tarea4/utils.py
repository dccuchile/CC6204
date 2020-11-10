import sys
from timeit import default_timer as timer

import torch
import numpy as np
from scipy.spatial import distance


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


def train4classification(net, train_loader, test_loader, optimizer, criterion, epochs=1, reports_every=1, device='cuda'):
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
      Y_logits, Y_probs = net(X)
      loss = criterion(Y_logits, Y)
      loss.backward()
      optimizer.step()

      # loss
      items = min(total_train, (i+1) * train_loader.batch_size)
      running_loss += loss.item()
      avg_loss = running_loss/(i+1)
      
      # accuracy
      max_prob, max_idx = torch.max(Y_probs, dim=1)
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
        Y_logits, Y_probs = net(X)
        max_prob, max_idx = torch.max(Y_probs, dim=1)
        running_acc += torch.sum(max_idx == Y).item()
        avg_acc = running_acc/total_test*100
      test_acc.append(avg_acc)
      sys.stdout.write(f', Val Acc:{avg_acc:02.2f}%.\n')
    else:
      sys.stdout.write('\n')

  return train_loss, train_acc, test_acc


def compute_ranks_x2y(x, y):
  dists = distance.cdist(x.cpu().detach().numpy(), y.cpu().detach().numpy(), 'euclidean')
  ranks = np.zeros(dists.shape[0])
  for i in range(len(ranks)):
    d_i = dists[i,:]
    inds = np.argsort(d_i)
    rank = np.where(inds == i)[0][0]
    ranks[i] = rank
  return ranks


def train4retrieval(img_net, text_net, train_loader, test_loader, optimizer, criterion, epochs=1, reports_every=1, device='cuda', norm=True):
  img_net.to(device)
  text_net.to(device)
  
  total_train = len(train_loader.dataset)
  total_test = len(test_loader.dataset)
  tiempo_epochs = 0
  train_loss, train_meanr, test_meanr = [], [], []

  for e in range(1,epochs+1):
    inicio_epoch = timer()

    # Aseguramos que todos los parámetros se entrenarán usando .train()
    img_net.train()
    text_net.train()

    # Variables para las métricas
    running_loss, running_meanr = 0.0, 0.0

    for i, data in enumerate(train_loader):
      # Desagregamos los datos y los pasamos a la GPU
      a, p, n = data
      if norm:
        a, p, n = l2norm(a), l2norm(p), l2norm(n)
      a, p, n = a.to(device), p.to(device), n.to(device)

      # Limpiamos los gradientes, pasamos el input por la red, calculamos
      # la loss, ejecutamos el backpropagation (.backward) 
      # y un paso del optimizador para modificar los parámetros
      optimizer.zero_grad()

      a_enc = img_net(a)
      p_enc = text_net(p)
      n_enc = text_net(n)

      loss = criterion(a_enc, p_enc, n_enc)
      loss.backward()
      optimizer.step()

      # loss
      items = min(total_train, (i+1) * train_loader.batch_size)
      running_loss += loss.item()
      avg_loss = running_loss/(i+1)

      # mean-rank
      ranks = compute_ranks_x2y(encoding, p)
      running_meanr += (ranks.mean()/len(a))
      avg_meanr = running_meanr/(i+1)

      # report
      sys.stdout.write(f'\rEpoch:{e}({items}/{total_train}), ' 
                       + f'Loss:{avg_loss:02.5f}, '
                       + f'Train mean-rank(normalized):{avg_meanr:02.3f}')

    if e % reports_every == 0:
      sys.stdout.write(', Validating...')
      train_loss.append(avg_loss)
      train_meanr.append(avg_meanr)

      img_net.eval()
      text_net.eval()

      running_meanr = 0.0
      for i, data in enumerate(test_loader):
        a, p, _ = data
        a, p = a.to(device), p.to(device)

        a_enc = img_net(a)
        p_enc = text_net(p)

        # mean-rank
        ranks = compute_ranks_x2y(a_enc, p_enc)
        running_meanr += (ranks.mean()/len(a))
        avg_meanr = running_meanr/(i+1)
      test_meanr.append(avg_meanr)
      sys.stdout.write(f', Val mean-rank(normalized):{avg_meanr:02.3f}.\n')
    else:
      sys.stdout.write('\n')

  return train_loss, train_meanr, test_meanr