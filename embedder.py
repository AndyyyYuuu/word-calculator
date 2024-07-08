
import numpy
import re
import random
import torch
import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
vocab_list = [word for word in list(tokenizer.vocab.keys()) if re.match(r'^[a-z]+$', word)]
random.shuffle(vocab_list)

def embed(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

    avg_embedding = torch.mean(embeddings[0], dim=0)
    return avg_embedding


def decode(vector):

    closest_word = None
    closest_distance = float('inf')

    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_vector = embed(word)
        distance = numpy.linalg.norm(word_vector - vector)
        if distance < closest_distance:
            closest_distance = distance
            closest_word = word
        if i % 10 == 0:
            print(f"{i}/{len(vocab_list)}: \"{closest_word}\"")

    return closest_word
