from trainer import train_word_embeddings
from VectorUtils import add_vectors
import math
import VectorUtils
import json
import codecs


class WordEmbedding:
    """
    Once you've trained a word embeddings object using the trainer, you can use this class for various analytical utils
    at the word level
    """

    def __init__(self, word_embedding_object=None, word_embedding_file=None):
        """
        if word_embedding_object is not None, word_embedding_file is ignored.
        :param word_embedding_object:
        :param word_embedding_file:
        """
        self._word_embedding_dict = dict()
        if word_embedding_object:
            self._word_embedding_dict = word_embedding_object
        elif word_embedding_file:
            with codecs.open(word_embedding_file, 'r', 'utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    for k, v in obj.items():
                        self._word_embedding_dict[k] = v
        else:
            raise Exception('Expected either a word embeddings file or a word embeddings object!')

    def get_similar_words(self, words, k=10, prune_threshold=1.0, print_warning=True):
        """

        :param words: either a single word or a list of words.
        :param k: Number of entries to retrieve.
        :param prune_threshold: if more than this fraction of elements in the vector are non-zero, then the
        word is probably a stop-word. Such words will not be considered in the similarity function. To disable,
        set to 1.0 (as by default). It is risky enabling this; we may deprecate in future.
        :param print_warning: if True, will print out warnings esp. when it doesn't find a word
        in embeddings dictionary. Disable at your own risk.
        :return: A list of k most similar words. A word may be 'multi-token' if you learned embeddings with max_n_grams
        > 1
        """
        list_of_words = list()
        # print type(words)
        if type(words) != list:
            if words in self._word_embedding_dict:
                list_of_words.append(words)
            else:
                if print_warning:
                    print 'Warning. Your word '+words+' is not in the embeddings dictionary. Returning None...'
                return None
        else:
            list_of_words = words
        results = dict()
        for seed_token in list_of_words:
            if seed_token not in self._word_embedding_dict:
                if print_warning:
                    print 'Warning. Your word '+seed_token+' is not in the embeddings dictionary. Skipping word...'
                continue
            scored_dict = self._generate_scored_dict(seed_token, prune_threshold=prune_threshold)
            results[seed_token] = WordEmbedding.extract_top_k(scored_dict, k=k)

        return results

    def get_vector(self, words, print_warning=True):
        """

        :param words: either a single word or a list of words.
        :param print_warning: if True (by default), it will print out a warning if it does not find a word
        in the embeddings dictionary. Disable at your own risk.
        :return: A single vector (or None).
        If there were multiple words, the vectors will be added. To normalize the vector use VectorUtils. You can
        modify the returned vector in any way without affecting the 'original' vectors in the embeddings dict.
        """
        if type(words) != list:
            if words in self._word_embedding_dict:
                return self._word_embedding_dict[words]
            else:
                if print_warning:
                    print 'Warning. Your word '+words+' is not in the embeddings dictionary. Returning None...'
                return None

        result = list()
        # We have a list of words
        for word in words:
            if word not in self._word_embedding_dict:
                if print_warning:
                    print 'Warning. Your word '+word+' is not in the embeddings dictionary. Moving on to next word...'
                continue
            result.append(list(self._word_embedding_dict[word]))
        if not result:
            if print_warning:
                print 'None of your words were in the embeddings dictionary. Returning None...'
            return None
        else:
            return add_vectors(result)

    def _generate_scored_dict(self, word, prune_threshold):
        scored_dict = dict()
        seed_vector = self._word_embedding_dict[word]
        for token, vector in self._word_embedding_dict.items():
            if token == word or VectorUtils.non_zero_element_fraction(vector) > prune_threshold:
                continue
            else:
                score = WordEmbedding.compute_abs_cosine_sim(seed_vector, vector)
                if score not in scored_dict:
                    scored_dict[score] = list()
                scored_dict[score].append(token)
        return scored_dict

    @staticmethod
    def compute_abs_cosine_sim(vector1, vector2):
        if len(vector1) != len(vector2):
            raise Exception
        total1 = 0.0
        total2 = 0.0
        sim = 0.0
        for i in range(0, len(vector1)):
            sim += (vector1[i]*vector2[i])
            total1 += (vector1[i]*vector1[i])
            total2 += (vector2[i]*vector2[i])
        total1 = math.sqrt(total1)
        total2 = math.sqrt(total2)
        if total1 == 0.0 or total2 == 0.0:
            print 'Divide by zero problem. Returning 0.0...'
            # print vector1
            # print vector2
            return 0.0
        else:
            return math.fabs(sim/(total1*total2))

    @staticmethod
    def extract_top_k(scored_results_dict, k, disable_k=False, reverse=True):
        """
        For internal use only. Do not invoke as user.
        :param scored_results_dict: a score always references a list
        :param k: Max. size of returned list.
        :param disable_k: ignore k, and sort the list by k
        :param reverse: if reverse is true, the top k will be the highest scoring k. If reverse is false,
        top k will be the lowest scoring k.
        :return:
        """
        count = 0
        # print k
        results = list()
        scores = scored_results_dict.keys()
        scores.sort(reverse=reverse)
        for score in scores:
            # print score
            # print count
            if count >= k and not disable_k:
                break
            vals = scored_results_dict[score]
            if disable_k:
                results += vals
                continue
            if count + len(vals) <= k:
                results += vals
                count = len(results)
            else:
                results += vals[0: k - count]
                count = len(results)
        # print results[0]
        return results
