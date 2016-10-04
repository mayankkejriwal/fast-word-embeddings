"""
(1) This code contains brief functions demonstrating some ways of using the code. These are NOT meant to be unittests.
(2) To try out the code snippets, manually change the paths below. For your convenience, all paths appear right at
the beginning of a function. All relevant files are included in the github repo.
(3) Example functions are in a 'natural' sequence. This is the typical sequence you would follow in your analytics,
starting by first putting your data in a compatible format, then training, then using the embeddings.
"""
from WordEmbedding import WordEmbedding
from DocEmbedding import DocEmbedding
import VectorUtils
import codecs
import json
from trainer import train_word_embeddings, train_doc_embeddings, train_annotation_models


def convert_jlines_to_compatible_format():
    """
    Takes a jlines file, such as what we see after extractions have been run on raw data, and assumes that
    the main text is in a field called 'high_recall_readability_text'. Outputs two files that can be respectively used
    for training word embeddings and doc embeddings.

    Be careful about newlines and whitespaces (a naive replacement of \n or \r is almost always insufficient). Since
    spaces etc. are not important to begin with, I just replace all whitespace-like characters with ' ' using the
    split utility.
    :return:
    """
    # input_file is actually a jlines
    input_file = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/' \
                 'fast-word-embeddings-datasets/part-00000-10lines.json'

    # these 'output' files should be input to the word embedding and doc embedding trainers respectively
    # doc ids are generated using a simple counter. In a 'real' application, a proper ID should be used instead.
    output_file_wordEmbedding = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/' \
                                'fast-word-embeddings-datasets/raw-lines.txt'
    output_file_docEmbedding = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/' \
                               'fast-word-embeddings-datasets/docids-raw-lines.txt'

    text_field = 'high_recall_readability_text'

    out_word = codecs.open(output_file_wordEmbedding, 'wb', 'utf-8')
    out_doc = codecs.open(output_file_docEmbedding, 'wb', 'utf-8')
    count = 1
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if text_field in obj and obj[text_field]:   # text_field should exist and be non-empty
                # string = obj[text_field].replace('\n', ' ').replace('\r', ' ') # not guaranteed to work!
                string = ' '.join(obj[text_field].split())
                string += '\n'
                out_word.write(string)
                out_doc.write(str(count)+'\t'+string)
            count += 1

    out_word.close()
    out_doc.close()


def word_doc_embedding_trainer_examples():
    """

    :return:
    """
    # input files to word and doc embeddings
    folder_path = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/fast-word-embeddings-datasets/'
    raw_text_file =  folder_path+'raw-lines.txt'
    doc_id_raw_text_file = folder_path+'docids-raw-lines.txt'

    # train embeddings. I'll write both out to file for posterity. If you don't write them out to file, you
    # can instantiate the resp. embedding class and use the write_to_file in those classes.
    word_embedding_obj = train_word_embeddings(raw_text_file, output_file=folder_path+'word_embedding_sample.jl')
    doc_embedding_obj = train_doc_embeddings(doc_id_raw_text_file, word_embedding_obj,
                                             output_file=folder_path+'doc_embedding_sample.jl')

    # instantiate embedding objects. To see what we can do with such objects, see the *_embedding_examples funcs below
    word_embedding = WordEmbedding(word_embedding_obj)
    doc_embedding = DocEmbedding(doc_embedding_obj)


def word_embedding_examples():
    """
    We show usage of various functions in WordEmbedding, as well as VectorUtils

    We assume a word embedding has been trained (using trainer.train_word_embeddings) and that the embedding
    has been written out to an output file. The use-case here is for human trafficking.
    :return: None
    """

    # for better performance, use word embeddings trained on bigger datasets e.g. unigram-part-00000-v2.json
    folder_path = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/' \
                 'fast-word-embeddings-datasets/'
    embeddings_file = folder_path+'word_embedding_sample.jl'

    # initialize
    embedding_obj = WordEmbedding(word_embedding_file=embeddings_file)

    print 'get some similar words...'
    print embedding_obj.get_similar_words('cleo')
    print embedding_obj.get_similar_words(['california', 'jessica', 'street', 'fake_word'], k=2)

    print '\nget (and print) vector for word/words...'
    vec1 = (embedding_obj.get_vector('cleo'))
    vec2 = (embedding_obj.get_vector(['california', 'jessica', 'street', 'fake_word']))
    print vec1
    print vec2

    print '\n(l2-)normalize and print these vectors...'
    print VectorUtils.normalize_vector(vec1)
    print VectorUtils.normalize_vector(vec2)

    print '\noriginal vectors are unmodified!!'
    print vec1
    print vec2

    print '\nto normalize many vectors, plug them into all into a list and invoke normalize_matrix instead...'
    matrix = list()
    matrix.append(vec1)
    matrix.append(vec2)
    print VectorUtils.normalize_matrix(matrix)

    print '\ncount fraction of non-zero elements in vectors. May or may not be useful for some other tasks...'
    print VectorUtils.non_zero_element_fraction(vec1)
    print VectorUtils.non_zero_element_fraction(vec2)


def doc_embedding_examples():
    """
    We use the doc embedding file that was trained in the traner.
    :return:
    """
    # initialize
    folder_path = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/' \
                      'fast-word-embeddings-datasets/'
    embeddings_file = folder_path+'doc_embedding_sample.jl'
    embedding_obj = DocEmbedding(doc_embedding_file=embeddings_file)

    print 'get some similar docs...'
    print embedding_obj.get_similar_docs('1')
    print embedding_obj.get_similar_docs(['1', '2', '15'], k=2)

    print '\nget (and print) vector for doc/docs...'
    vec1 = (embedding_obj.get_vector('1'))
    vec2 = (embedding_obj.get_vector(['1', '2', '15']))
    print vec1
    print vec2


def annotation_trainer_example():
    """
    We use a pre-trained embedding file (the unigrams file that was trained on 1 GB data, not the
    sample generated in trainer_examples),  and an annotated file from our ground-truth to output models.
    Note that these models must be subsequently used in your extractors. We do not provide functions for testing
    the models in this piece of code, so it's possible there are 'semantic' (as opposed to syntactic) mistakes.
    Our annotations classifiers are still highly experimental.
    :return:
    """
    # declare paths
    folder_path = '/Users/mayankkejriwal/ubuntu-vm-stuff/home/mayankkejriwal/tmp/' \
                  'fast-word-embeddings-datasets/'
    annotated_jlines_file = folder_path+'annotated-cities.jl'
    word_embedding_file = folder_path+'unigram-part-00000-v2.json'
    classification_model_output_file = folder_path+'classification_model/classification_model'
    feature_model_output_file = folder_path+'feature_model/feature_model'
    train_annotation_models(annotated_jlines_file=annotated_jlines_file,
        text_attribute='high_recall_readability_text', annotated_attribute='annotated_cities',
        correct_attribute='correct_cities', word_embedding_object=None,
        classification_model_output_file=classification_model_output_file,
        feature_model_output_file=feature_model_output_file,
        word_embedding_file=word_embedding_file)


# annotation_trainer_example()