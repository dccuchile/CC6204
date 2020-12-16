import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def extract_text_from_set(data):
    sentences = []
    for element in tqdm(data):
        for captions in element[1]:
            sentences.append(captions)            
    return sentences

def tokenize_text(texts, tokenizer, counter=None, sos_token='<sos>', end_token='.'):
    tokenized_texts = []
    for text in texts:
        tokenized_text = tokenizer(text.lower())
        if counter != None:
            counter.update(tokenized_text)
        if tokenized_text[-1] != end_token:
            tokenized_text.append(end_token)
        tokenized_text.insert(0, sos_token)
        tokenized_texts.append(tokenized_text)
    return tokenized_texts, counter

def encode_sentences(text_tokens, vocab, encoder):
    encoded_tokens = []
    for sentence in text_tokens:
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

def pad_sequence_with_lengths(data, pad_idx):
    sequences = torch.nn.utils.rnn.pad_sequence([d[0] for d in data], padding_value=pad_idx)
    seq_lens = torch.tensor([d[1] for d in data], dtype=torch.long)
    return sequences, seq_lens

def train_one_epoch(model, voc_size, dataloader, optimizer, loss_fn, device):
    running_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        D, Lengths = batch
        D = D.to(device)
        X, Y = D[:-1, :], D[1:, :]
        logits = model(X)
        logits = logits.view(-1, voc_size)
        target = Y.view(-1)
        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    mean_loss = running_loss / (idx + 1)
    return mean_loss

def eval_one_epoch(model, voc_size, dataloader, loss_fn, device):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            D, Lengths = batch
            D = D.to(device)
            X, Y = D[:-1, :], D[1:, :]
            logits = model(X)
            logits = logits.view(-1, voc_size)
            target = Y.view(-1)
            loss = loss_fn(logits, target)
            running_loss += loss.item()
    mean_loss = running_loss / (idx + 1)
    return mean_loss

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
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.init_word = init_word
        self.stop_word = stop_word

    def original_image(self, idx):
        return self.data[idx][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        captions = encode_captions(d[1], self.tokenizer, self.encoder, self.init_word, self.stop_word)
        # elige siempre la primera caption
        caption = captions[0]
        sequence = torch.tensor(caption, dtype=torch.long)
        image = self.transform(d[0])
        return sequence, len(sequence), image

def pad_sequence_with_images(data, pad_idx):
    sequences = torch.nn.utils.rnn.pad_sequence([d[0] for d in data], padding_value=pad_idx)
    seq_lens = torch.tensor([d[1] for d in data], dtype=torch.long)
    images = torch.stack([d[2] for d in data])
    return sequences, seq_lens, images
