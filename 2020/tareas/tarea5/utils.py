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