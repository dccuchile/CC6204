from tqdm import tqdm

from torch.utils.data import Dataset


def extract_text_from_set(data):
    sentences = []
    for element in tqdm(data):
        sentences.append([sentence for sentence in element[1]])            
    return sentences


def tokenize_text(texts, tokenizer, counter):
    tokenized_texts = []
    for text in tqdm(texts):
        tokenized_text = tokenizer(text.lower())
        counter.update(tokenized_text)
        if tokenized_text[-1] != ".":
            tokenized_text.append(".")
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
        self.n = len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return self.n


def train_one_epoch(model, dataloader, optimizer, loss_fun, clip_value, device, num_classes=None):
    running_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        x, sl = batch
        x, sl = x.to(device), sl.to(device)
        if num_classes is not None:
            one_hot = torch.nn.functional.one_hot(x, num_classes).float()
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


def eval_one_epoch(model, dataloader, loss_fun, device, num_classes=None):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, sl = batch
            x, sl = x.to(device), sl.to(device)
            if num_classes is not None:
                one_hot = torch.nn.functional.one_hot(x, num_classes).float()
                logits = model(one_hot[:, :-1])
            loss = loss_fun(logits.transpose(1, 2), x[:, 1:])
            running_loss += loss.item()
    running_loss /= (idx + 1)
    return running_loss


def plot_loss(loss):
    plt.figure()
    fig = plt.plot(losses)
    plt.legend(fig, ["train", "val"])
    plt.xlabel("epoch")
    plt.ylabel("cross entropy")
    return