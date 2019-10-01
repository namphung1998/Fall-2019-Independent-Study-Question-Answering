import torchtext

def get_embedding(counter, limit=-1, vec_size=300):
    print(f'Initializing GloVe embeddings...')
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    glove = torchtext.vocab.GloVe(name='6B', dim=vec_size, max_vectors=len(filtered_elements))