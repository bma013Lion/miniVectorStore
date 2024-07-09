from vector_store import VectorStore
import numpy as np

# create a vector store instance
vector_store = VectorStore()

# define your sentences
sentences = [
    "I eat mangos",
    "I love mangos",
    "mangos are my favorite fruits",
    "but they are not good for my health",
    "because mangos contain a lot of sugar"
]

# Tokenization and vocabulary building
vocabulary = set()

for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocab
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}
for sentence in sentences: 
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    
    sentence_vectors[sentence] = vector

# Add vectors to the vector store
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# searching for similarity
query_sentence = "Mango is the best fruit"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()
for tokns in query_tokens:
    query_vector[word_to_index[token]] += 1

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)
print("Query Sentence: ", query_sentence)
print("Similar Sentence: ")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity =  {similarity: .4f}")


