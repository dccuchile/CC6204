def extract_text_from_set(data):
    sentences = []
    for element in tqdm(data):
        for sentence in element[1]:
            sentences.append(sentence)
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