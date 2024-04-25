import random

from nltk import ngrams
from nltk.tag import hmm
from nltk.probability import FreqDist
import os
from IPython.display import Image
import graphviz
from hmmlearn import hmm
from nltk import ngrams
import numpy as np

# Function to train an N-gram language model
def train_ngram_model(tokens, n=2):
    ngrams_list = list(ngrams(tokens, n))
    ngram_model = FreqDist(ngrams_list)
    return ngram_model


# Function to generate text using the trained model
def generate_text(model, seed, length=20):
    # n is the number of words in the ngram window
    n = len(list(model.keys())[0])
    current = seed.split()[-n + 1:]
    result = seed.split()

    for _ in range(length):
        # TODO
        # predict the next word based on the most probable ngram.
        next_word = random.choice([word for word, freq in model.items() if word[:-1] == tuple(current)])
        result.append(next_word[-1])
        current = result[-n + 1:]

    return ' '.join(result)


# Function to train HMM and draw the final graph using graphviz for a specific node
def train_and_draw_hmm_for_node(tokens, n_gram_order, initial_node):
    # Generate n-grams using nltk
    n_grams = list(ngrams(tokens, n_gram_order))

    # Create a mapping of n-grams to integer indices
    ngram_index = {ngram: idx for idx, ngram in enumerate(set(n_grams))}

    # Convert the corpus to integer indices
    corpus_indices = [ngram_index[ngram] for ngram in n_grams]

    # Reshape the data to fit HMM requirements
    X = np.array(corpus_indices).reshape(-1, 1)

    # Train the Hidden Markov Model
    model = hmm.MultinomialHMM(n_components=len(ngram_index), n_iter=100)
    model.fit(X)

    # Create a directed graph using graphviz
    dot = graphviz.Digraph(comment='Hidden Markov Model')

    # Add the initial node to the graph
    initial_idx = ngram_index[initial_node]
    dot.node(str(initial_idx), label=' '.join(initial_node))

    # Add outgoing edges with weights for the initial node
    for to_state, prob in enumerate(model.transmat_[initial_idx]):
        to_ngram = list(ngram_index.keys())[to_state]
        dot.node(str(to_state), label=' '.join(to_ngram))
        dot.edge(str(initial_idx), str(to_state), label=f"{prob:.2f}")

    # Save the graph to a file
    dot.render('hmm_graph', format='png', cleanup=True)

    # Display the image in the notebook
    image_path = 'hmm_graph.png'
    display_image = Image(image_path)

    # Delete the image file after displaying
    os.remove(image_path)

    return display_image