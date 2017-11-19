#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.metrics as metrics
from tensorflow.python.platform import gfile
from tensorflow.contrib import lookup
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# variables set by init()
BUCKET = None
TRAIN_STEPS = 400
WORD_VOCAB_FILE = None
N_WORDS = -1

# hardcoded into graph
BATCH_SIZE = 16

# describe your data
TARGETS = ['1', '0']
MAX_DOCUMENT_LENGTH = 77838
CSV_COLUMNS = ['reviews', 'inspection_result', 'categories', 'zip_code',  'review_count', 'avg_rating']
LABEL_COLUMN = 'inspection_result'
DEFAULTS = [['null'], ['null'], ['null'], ['null'], [0.0], [2.5]]
PADWORD = 'ZYXW'


def init(bucket, num_steps):
    global BUCKET, TRAIN_STEPS, WORD_VOCAB_FILE, N_WORDS
    BUCKET = bucket
    TRAIN_STEPS = num_steps
    WORD_VOCAB_FILE = 'gs://{}/data/vocab_words'.format(BUCKET)
    N_WORDS = save_vocab('gs://{}/data/train.csv'.format(BUCKET), 'reviews', WORD_VOCAB_FILE)


def save_vocab(trainfile, txtcolname, outfilename):
    if trainfile.startswith('gs://'):
        import subprocess
        tmpfile = "vocab.csv"
        subprocess.check_call("gsutil cp {} {}".format(trainfile, tmpfile).split(" "))
        filename = tmpfile
    else:
        filename = trainfile
    import pandas as pd
    df = pd.read_csv(filename, header=None, sep='|', names=['reviews', 'inspection_result', 'categories', 'zip_code',  'review_count', 'avg_rating'])

    # the text to be classified
    vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=5)
    vocab_processor.fit(df[txtcolname])


    with gfile.Open(outfilename, 'wb') as f:
        f.write("{}\n".format(PADWORD))
        for word, index in vocab_processor.vocabulary_._mapping.iteritems():
            f.write("{}\n".format(word))
    nwords = len(vocab_processor.vocabulary_)
    print('{} words into {}'.format(nwords, outfilename))
    return nwords + 2  # PADWORD and <UNK>


def read_dataset(prefix):
    # use prefix to create filename
    filename = 'gs://{}/data/{}*csv*'.format(BUCKET, prefix)
    if prefix == 'train':
        mode = tf.contrib.learn.ModeKeys.TRAIN
    else:
        mode = tf.contrib.learn.ModeKeys.EVAL

    # the actual input function passed to TensorFlow
    def _input_fn():
        # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)

        # read CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS, field_delim='|')
        features = dict(zip(CSV_COLUMNS, columns))

        label = features.pop(LABEL_COLUMN)

        # make targets numeric
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
        target = table.lookup(label)
        print(features,target)
        return features, target
    return _input_fn


# CNN model parameters
EMBEDDING_SIZE = 10
WINDOW_SIZE = EMBEDDING_SIZE
STRIDE = int(WINDOW_SIZE / 2)
STRIDE = 2


def cnn_model(features, target, mode):
    table = lookup.index_table_from_file(vocabulary_file=WORD_VOCAB_FILE, num_oov_buckets=1, default_value=-1)

    # string operations
    reviews = tf.squeeze(features['reviews'], [1])
    words = tf.string_split(reviews, delimiter=" ")
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(densewords)
    print(numbers)
    padding = tf.constant([[0, 0], [0, MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0, 0], [-1, MAX_DOCUMENT_LENGTH])
    print('words_sliced={}'.format(words))


    # layer to take the words and convert them into vectors (embeddings)
    embeds = tf.contrib.layers.embed_sequence(sliced, vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)

    print('words_embed={}'.format(embeds))

    # now do convolution
    conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE, stride=STRIDE, padding='SAME')

    # RELU
    transfnonlin = tf.nn.relu(conv)

    # MAXPOOL
    maxpool = tf.layers.max_pooling1d(transfnonlin, WINDOW_SIZE, strides=STRIDE, padding='SAME')

    words = tf.squeeze(maxpool, [2])
    print('words_conv={}'.format(words))

    n_classes = len(TARGETS)

    #categories
    CATEGORIES_LIST = tf.constant(['Salvadoran', 'Tapas/Small Plates', 'Fondue', 'Buffets', 'Gluten-Free', 'Sandwiches', 'Creperies', 'Sushi Bars', 'Belgian', 'Dim Sum', 'French', 'Italian', 'Haitian', 'Salad', 'Persian/Iranian', 'Restaurants', 'Steakhouses', 'Mediterranean', 'Moroccan', 'Delis', 'Puerto Rican', 'Halal', 'Cantonese', 'Senegalese', 'Polish', 'Tex-Mex', 'Caribbean', 'American (New)', 'Breakfast & Brunch', 'Food Stands', 'Indonesian', 'Hawaiian', 'Cheesesteaks', 'Tapas Bars', 'Szechuan', 'Thai', 'Fish & Chips', 'Cuban', 'Cajun/Creole', 'Cafes', 'Scandinavian', 'Greek', 'African', 'Chinese', 'Burgers', 'Asian Fusion', 'Ethiopian', 'Turkish', 'Middle Eastern', 'Live/Raw Food', 'Indian', 'Latin American', 'Brazilian', 'Chicken Wings', 'Korean', 'Pakistani', 'Barbeque', 'Shanghainese', 'Southern', 'Vegan', 'Diners', 'Russian', 'Gastropubs', 'Afghan', 'Vegetarian', 'Malaysian', 'Hot Pot', 'Kosher', 'Modern European', 'Irish', 'German', 'Taiwanese', 'Laotian', 'Mongolian', 'Basque', 'Vietnamese', 'Scottish', 'Fast Food', 'Australian', 'Venezuelan', 'Colombian', 'Pizza', 'Filipino', 'Egyptian', 'Trinidadian', 'Himalayan/Nepalese', 'Food Court', 'Lebanese', 'Seafood', 'British', 'Soup', 'Comfort Food', 'Mexican', 'American (Traditional)', 'Japanese', 'Cambodian', 'Hot Dogs', 'Spanish', 'Soul Food'])
    table_cat = tf.contrib.lookup.index_table_from_tensor(mapping=CATEGORIES_LIST, num_oov_buckets=1, default_value=-1)
    categories_string = tf.squeeze(features['categories'], [1])
    categories_words = tf.string_split(categories_string, delimiter=",")
    categories_dense_words = tf.sparse_tensor_to_dense(categories_words, default_value=PADWORD)
    category_ids = table_cat.lookup(categories_dense_words)
    padding_cat = tf.constant([[0, 0], [0, 100]])
    padded_cat = tf.pad(category_ids, padding_cat)
    sliced_cat = tf.slice(padded_cat, [0, 0], [-1, 100])

    embed_categories = tf.contrib.layers.embed_sequence(sliced_cat, vocab_size=100, embed_dim=EMBEDDING_SIZE)
    print("embed_categories shape")
    print(embed_categories.get_shape())

    categories_out = tf.contrib.layers.fully_connected(embed_categories, 1)

    print("categories_out shape")
    print(categories_out.get_shape())

    categories = tf.squeeze(categories_out, [2])

    #zip_codes
    ZIP_CODE_LIST = tf.constant( ['98101', '98102', '98103', '98104', '98105', '98106', '98107',
         '98108', '98109', '98112', '98115', '98116', '98117', '98118',
         '98119', '98121', '98122', '98125', '98126', '98133', '98134',
         '98136', '98144', '98146', '98166', '98168', '98177', '98178',
         '98188', '98199'])
    table_zip = tf.contrib.lookup.index_table_from_tensor(mapping=ZIP_CODE_LIST, num_oov_buckets=1, default_value=-1)
    zip_string = tf.squeeze(features['zip_code'], [1])
    zip_id = table_zip.lookup(zip_string)
    zip_one_hot = tf.one_hot(zip_id, 30, 1.0, 0.0)
    #padding_zip = tf.constant([[0, 0], [0, 30]])
    #padded_zip = tf.pad(zip_one_hot, padding_zip)
    #sliced_zip = tf.slice(padded_zip, [0, 0], [-1, 30])

    ##print("sliced_zip shape")
    #print(sliced_zip.get_shape())

    #zip_out = tf.contrib.layers.fully_connected(zip_one_hot, 1)


    zip = tf.contrib.layers.fully_connected(zip_one_hot, 1)
    
    #print("zip_out shape")
    #print(zip_out.get_shape())

    #zip = tf.squeeze(zip_out, [2])

    #continuous variables
    avg_rating = features['avg_rating']
    review_count = features['review_count'] #tf.string_to_number(

    print(avg_rating)
    print(review_count)

    norm_avg_review = tf.nn.l2_normalize(avg_rating, dim=0)
    norm_review_count = tf.nn.l2_normalize(review_count, dim=0)

    print("Shape of norm_avg_review:")
    print(norm_avg_review)

    print("Shape of norm_review_count:")
    print(norm_review_count)

    print("Shape of categories:")
    print(categories)

    num_combined = tf.concat([norm_avg_review, norm_review_count], 1)

    nums = tf.contrib.layers.fully_connected(num_combined, 1)

    combined = tf.concat([nums, categories, words, zip], 1)

    logits = tf.contrib.layers.fully_connected(combined, n_classes, activation_fn=None)

    predictions_dict = {
        'source': tf.gather(TARGETS, tf.argmax(logits, 1)),
        'class': tf.argmax(logits, 1),
        'prob': tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
        train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            learning_rate=0.01)
    else:
        loss = None
        train_op = None

    return tflearn.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op)


def serving_input_fn():
    feature_placeholders = {
        'reviews': tf.placeholder(tf.string, [None]),
        'categories': tf.placeholder(tf.string, [None]),
        'zip_code': tf.placeholder(tf.string, [None]),
        'review_count': tf.placeholder(tf.float32, [None]),
        'avg_rating': tf.placeholder(tf.float32, [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tflearn.utils.input_fn_utils.InputFnOps(
        features,
        None,
        feature_placeholders)


def get_train():
    return read_dataset('train')


def get_valid():
    return read_dataset('eval')


from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


def metric_fn(predictions=None, labels=None, weights=None):
    P, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    R, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    eps = 1e-5
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))


def experiment_fn(output_dir):
    # run experiment
    return tflearn.Experiment(
        tflearn.Estimator(model_fn=cnn_model, model_dir=output_dir),
        train_input_fn=get_train(),
        eval_input_fn=get_valid(),
        eval_metrics={
            "accuracy": tf.contrib.learn.MetricSpec(
                 metric_fn=metrics.streaming_accuracy,
                 prediction_key='class'),
            "precision": tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key='class'),
            "recall": tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key='class'),
            "f1score": tf.contrib.learn.MetricSpec(
                metric_fn=metric_fn,
                prediction_key='class')
        },
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        train_steps=TRAIN_STEPS
    )


