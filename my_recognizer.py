import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for v in test_set.sentences_index:
        for test_word in test_set.sentences_index[v]:
            probability = {}
            for model_word in models:
                model = models[model_word]
                X, lengths = test_set.get_item_Xlengths(test_word)
                try:
                    probability[model_word] = model.score(X, lengths)
                except:
                    probability[model_word] = -1000000
            probabilities.append(probability)
    for probability in probabilities:
        guesses.append(max(probability.items(), key=lambda x: x[1])[0])
    return probabilities, guesses

