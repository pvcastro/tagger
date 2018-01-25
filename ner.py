import os
import numpy as np
import itertools
import loader
import time

from utils import create_input, models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping, update_tag_scheme, prepare_dataset, augment_with_pretrained
from model import Model


class NER(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        # Check parameters validity
        assert os.path.isfile(parameters['train'])
        assert os.path.isfile(parameters['dev'])
        assert os.path.isfile(parameters['test'])
        assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
        assert 0. <= parameters['dropout'] < 1.0
        assert parameters['tag_scheme'] in ['iob', 'iobes']
        assert not parameters['all_emb'] or parameters['pre_emb']
        assert not parameters['pre_emb'] or parameters['word_dim'] > 0
        assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])
        # Check evaluation script / folders
        if not os.path.isfile(eval_script):
            raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
        if not os.path.exists(eval_temp):
            os.makedirs(eval_temp)
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        self.parameters = parameters

    def train(self, n_epochs=100, freq_eval=1000, verbose=True, eval_test_set=False):
        """
        :param n_epochs: number of epochs over the training set
        :param freq_eval: evaluate on dev every freq_eval steps
        :return: Saves the model with the best F1-Score, evaluated on the dev set
        """
        # Initialize model
        model = Model(parameters=self.parameters, models_path=models_path)
        print("Model location: %s" % model.model_path)

        # Data parameters
        lower = self.parameters['lower']
        zeros = self.parameters['zeros']
        tag_scheme = self.parameters['tag_scheme']

        # Load sentences
        train_sentences = loader.load_sentences(self.parameters['train'], lower, zeros)
        dev_sentences = loader.load_sentences(self.parameters['dev'], lower, zeros)
        test_sentences = loader.load_sentences(self.parameters['test'], lower, zeros)

        # Use selected tagging scheme (IOB / IOBES)
        update_tag_scheme(train_sentences, tag_scheme)
        update_tag_scheme(dev_sentences, tag_scheme)
        update_tag_scheme(test_sentences, tag_scheme)

        # Create a dictionary / mapping of words
        # If we use pretrained embeddings, we add them to the dictionary.
        if self.parameters['pre_emb']:
            dico_words_train = word_mapping(train_sentences, lower)[0]
            dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                self.parameters['pre_emb'],
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in dev_sentences + test_sentences])
                ) if not self.parameters['all_emb'] else None
            )
        else:
            dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
            dico_words_train = dico_words

        # Create a dictionary and a mapping for words / POS tags / tags
        dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        # Index data
        train_data = prepare_dataset(
            train_sentences, word_to_id, char_to_id, tag_to_id, lower
        )
        dev_data = prepare_dataset(
            dev_sentences, word_to_id, char_to_id, tag_to_id, lower
        )
        test_data = prepare_dataset(
            test_sentences, word_to_id, char_to_id, tag_to_id, lower
        )

        print("%i / %i / %i sentences in train / dev / test." % (
            len(train_data), len(dev_data), len(test_data)))

        # Save the mappings to disk
        print('Saving the mappings to disk...')
        model.save_mappings(id_to_word, id_to_char, id_to_tag)

        # Build the model
        f_train, f_eval = model.build(**self.parameters)

        # Reload previous model values
        if self.parameters['reload']:
            print('Reloading previous model...')
            model.reload()

        #
        # Train network
        #
        singletons = set([word_to_id[k] for k, v
                          in dico_words_train.items() if v == 1])
        best_dev = -np.inf
        best_test = -np.inf
        count = 0
        for epoch in range(n_epochs):
            epoch_costs = []
            print("Starting epoch %i at..." % epoch, time.ctime())
            for i, index in enumerate(np.random.permutation(len(train_data))):
                count += 1
                input = create_input(train_data[index], self.parameters, True, singletons)
                new_cost = f_train(*input)
                epoch_costs.append(new_cost)
                if i % 50 == 0 and i > 0 == 0 and verbose:
                    print("%i, cost average: %f" % (i, np.mean(epoch_costs[-50:])))
                if count % freq_eval == 0:
                    dev_score = evaluate(self.parameters, f_eval, dev_sentences,
                                         dev_data, id_to_tag, verbose=verbose)
                    if eval_test_set:
                        test_score = evaluate(self.parameters, f_eval, test_sentences,
                                              test_data, id_to_tag, verbose=verbose)
                    print("Score on dev: %.5f" % dev_score)
                    if eval_test_set:
                        print("Score on test: %.5f" % test_score)
                    if dev_score > best_dev:
                        best_dev = dev_score
                        print("New best score on dev.")
                        print("Saving model to disk...")
                        model.save()
                    if eval_test_set:
                        if test_score > best_test:
                            best_test = test_score
                            print("New best score on test.")
            print("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))
        return best_dev
