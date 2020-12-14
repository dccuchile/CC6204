from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical


def extract_text_from_set(data):
    sentences = []
    for element in tqdm(data):
        for captions in element[1]:
            sentences.append(captions)            
    return sentences


def tokenize_text(texts, tokenizer, counter):
    tokenized_texts = []
    for text in tqdm(texts):
        tokenized_text = tokenizer(text.lower())
        counter.update(tokenized_text)
        if tokenized_text[-1] != ".":
            tokenized_text.append(".")
        tokenized_text.insert(0, "<sos>")
        tokenized_texts.append(tokenized_text)
    return tokenized_texts, counter


def encode_sentences(text_tokens, vocab, encoder):
    encoded_tokens = []
    for sentence in tqdm(text_tokens):
        encoded_tokens.append([encoder[word] for word in sentence])
    return encoded_tokens


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = [torch.tensor(text, dtype=torch.long) for text in texts]
        self.seq_len = [len(text) for text in texts]
        self.n = len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.seq_len[index]

    def __len__(self):
        return self.n


def pad_sequence(x, pad_idx):
    sequences = torch.nn.utils.rnn.pad_sequence([xi[0] for xi in x], batch_first=True, padding_value=pad_idx)
    seq_len = torch.tensor([xi[1] for xi in x], dtype=torch.long)
    return sequences, seq_len


def train_one_epoch(model, dataloader, optimizer, loss_fun, clip_value, device):
    running_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        x, sl = batch
        x, sl = x.to(device), sl.to(device)
        if model.emb_flag:
            logits = model(x[:, :-1])
        else:            
            one_hot = torch.nn.functional.one_hot(x, model.nout).float()
            logits = model(one_hot[:, :-1])
        loss = loss_fun(logits.transpose(1, 2), x[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        running_loss += loss.item()
    running_loss /= (idx + 1)
    return running_loss


def eval_one_epoch(model, dataloader, loss_fun, device):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, sl = batch
            x, sl = x.to(device), sl.to(device)
            if model.emb_flag:
                logits = model(x[:, :-1])
            else:            
                one_hot = torch.nn.functional.one_hot(x, model.nout).float()
                logits = model(one_hot[:, :-1])
            loss = loss_fun(logits.transpose(1, 2), x[:, 1:])
            running_loss += loss.item()
    running_loss /= (idx + 1)
    return running_loss


def plot_loss(loss):
    plt.figure()
    fig = plt.plot(loss)
    plt.legend(fig, ["train", "val"])
    plt.xlabel("epoch")
    plt.ylabel("cross entropy")
    return


def generate_sentence(model, init_word, stop_word, encoder, vocab):
    sentence = []
    model.to("cpu")
    model.eval()
    state = torch.zeros(model.nlayers, 1, model.nh)
    next_word = encoder[init_word]
    x = torch.tensor([next_word], dtype=torch.long).view(1, 1)
    if not model.emb_flag:
        x = torch.nn.functional.one_hot(x, model.nout).float()        
    sentence.append(next_word)
    with torch.no_grad():
        while next_word != encoder[stop_word]:
            logits, state = model.evaluate(x, state)
            next_word = Categorical(logits=logits).sample().item()
            sentence.append(next_word)
            x = torch.tensor([next_word], dtype=torch.long).view(1, 1)
            if not model.emb_flag:
                x = torch.nn.functional.one_hot(x, model.nout).float() 
    sentence = [vocab[idx] for idx in sentence]
    sentence = " ".join(sentence)
    return sentence


def encode_captions(sentences, tokenizer, encoder, init_word, stop_word):
    encoded_sentences = []
    for sentence in sentences:
        encoded_sentence = [encoder[word] for word in tokenizer(sentence.lower())]
        if encoded_sentence[-1] != encoder[stop_word]:
            encoded_sentence.append(encoder[stop_word])
        encoded_sentence.insert(0, encoder[init_word])
        encoded_sentences.append(encoded_sentence)
    return encoded_sentences


class CaptioningDataset(Dataset):
    def __init__(self, data, transform, tokenizer, encoder, init_word, stop_word):
        captions = [encode_captions(xi[1], tokenizer, encoder, init_word, stop_word) for xi in data]
        self.sequences = []
        for caption in captions:
            self.sequences.append([torch.tensor(sentence, dtype=torch.long) for sentence in caption])
        self.original_image = [xi[0] for xi in data]
        self.images = [transform(image) for image in self.original_image]
        self.n = len(self.sequences)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sequence = self.sequences[idx][0]
        seq_len = len(sequence)
        return sequence, seq_len, self.images[idx]


def pad_sequence_with_images(x, pad_idx):
    sequences = torch.nn.utils.rnn.pad_sequence([xi[0] for xi in x], batch_first=True, padding_value=pad_idx)
    seq_len = torch.tensor([xi[1] for xi in x], dtype=torch.long)
    images = torch.stack([xi[2] for xi in x])
    return sequences, seq_len, images


def train_one_epoch_captioning(model, dataloader, optimizer, loss_fun, clip_value, device):
    running_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        x, sl, img = batch
        x, sl, img = x.to(device), sl.to(device), img.to(device)
        if model.emb_flag:
            logits = model(x[:, :-1], img)
        else:            
            one_hot = torch.nn.functional.one_hot(x, model.nout).float()
            logits = model(one_hot[:, :-1], img)
        loss = loss_fun(logits.transpose(1, 2), x[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        running_loss += loss.item()
    running_loss /= (idx + 1)
    return running_loss


def eval_one_epoch_captioning(model, dataloader, loss_fun, device):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, sl, img = batch
            x, sl, img = x.to(device), sl.to(device), img.to(device)
            if model.emb_flag:
                logits = model(x[:, :-1], img)
            else:            
                one_hot = torch.nn.functional.one_hot(x, model.nout).float()
                logits = model(one_hot[:, :-1], img)
            loss = loss_fun(logits.transpose(1, 2), x[:, 1:])
            running_loss += loss.item()
    running_loss /= (idx + 1)
    return running_loss


def show_image(image):
    mu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inverse_trans = transforms.Normalize(mean=- mu / std, std=1 / std)
    numpy_image = inverse_trans(image).permute(1, 2, 0).cpu().numpy()
    numpy_image *= 255
    plt.figure()
    plt.imshow(numpy_image.astype(np.uint8))
    return


def generate_caption(model, image, stop_word, encoder, vocab):
    sentence = []
    model.to("cpu")
    model.eval()
    state = model.cnn_model(image.unsqueeze(0)).repeat(model.rnn_model.nlayers, 1, 1)
    next_word = encoder["<sos>"]
    x = torch.tensor([next_word], dtype=torch.long).view(1, 1)
    if not model.emb_flag:
        x = torch.nn.functional.one_hot(x, model.nout).float()        
    sentence.append(next_word)
    with torch.no_grad():
        while next_word != encoder[stop_word]:
            logits, state = model.evaluate(x, state)
            next_word = Categorical(logits=logits).sample().item()
            sentence.append(next_word)
            x = torch.tensor([next_word], dtype=torch.long).view(1, 1)
            if not model.emb_flag:
                x = torch.nn.functional.one_hot(x, model.nout).float() 
    sentence = [vocab[idx] for idx in sentence]
    sentence = sentence[1:]
    sentence = " ".join(sentence)
    return sentence