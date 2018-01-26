#!/usr/bin/env python

import time
import numpy as np
from loader import prepare_sentence
from utils import create_input, iobes_iob, zero_digits
from model import Model


def tag(model, line):
    # Load existing model
    print("Loading model...")
    model = Model(model_path=model)
    parameters = model.parameters

    # Load reverse mappings
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
    ]

    # Load the model
    _, f_eval = model.build(training=False, **parameters)
    model.reload()

    start = time.time()

    print('Tagging...')
    words_ini = line.rstrip().split()

    # Replace all digits with zeros
    if parameters['zeros']:
        line = zero_digits(line)
    words = line.rstrip().split()
    # Prepare input
    sentence = prepare_sentence(words, word_to_id, char_to_id, lower=parameters['lower'])
    input = create_input(sentence, parameters, False)
    # Decoding
    if parameters['crf']:
        y_preds = np.array(f_eval(*input))[1:-1]
    else:
        y_preds = f_eval(*input).argmax(axis=1)
    y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
    # Output tags in the IOB2 format
    if parameters['tag_scheme'] == 'iobes':
        y_preds = iobes_iob(y_preds)
    # Write tags
    assert len(y_preds) == len(words)

    print('---- sentence tagged in %.4fs ----' % (time.time() - start))

    return ' '.join(w + '__' + str(y) for w, y in zip(words_ini, y_preds))
