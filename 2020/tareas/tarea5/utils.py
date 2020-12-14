def extract_text_from_set(data):
    sentences = []
    for element in tqdm(data):
        for sentence in element[1]:
            sentences.append(sentence)
    return sentences