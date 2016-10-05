from WordEmbedding import WordEmbedding
from VectorUtils import add_vectors
import json
import codecs


class DocEmbedding:
    """
    Once you've trained a doc embeddings object using the trainer, you can use this class for various analytical utils
    at the doc level.

    Note that this class is more 'experimental' (in other words, riskier) than the word embedding
    class. If you want to go the safe route, using tf-idf/jaccard/lsh may be more appropriate. This class is
    particularly sensitive to the 'word_blacklist' argument in doc_embedding trainer
    """

    def __init__(self, doc_embedding_object=None, doc_embedding_file=None):
        """
        if doc_embedding_object is not None, doc_embedding_file is ignored.
        :param doc_embedding_object:
        :param doc_embedding_file:
        """
        self._doc_embedding_dict = dict()
        if doc_embedding_object:
            self._doc_embedding_dict = doc_embedding_object
        elif doc_embedding_file:
            with codecs.open(doc_embedding_file, 'r', 'utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    for k, v in obj.items():
                        self._doc_embedding_dict[k] = v
        else:
            raise Exception('Expected either a doc embeddings file or a doc embeddings object!')

    def write_embeddings_to_file(self, output_file):
        """

        :param output_file:
        :return: None
        """
        out = codecs.open(output_file, 'w', 'utf-8')
        for k, v in self._doc_embedding_dict.items():
            answer = dict()
            answer[k] = v
            json.dump(answer, out)
            out.write('\n')
        out.close()

    def get_similar_docs(self, doc_ids, k=10, print_warning=True):
        """

        :param doc_ids: either a single doc_id or a list of doc_ids
        :param k: number of similar results to return
        :param print_warning: if True (by default), it will print out a warning if it does not find a docid
        in the embeddings dictionary. Disable at your own risk.
        :return: A list of doc_ids
        """
        list_of_docids = list()
        if type(doc_ids) != list:
            if doc_ids in self._doc_embedding_dict:
                list_of_docids.append(doc_ids)
            else:
                if print_warning:
                    print 'Warning. Your docid ' + doc_ids + ' is not in the embeddings dictionary. Returning None...'
                return None
        else:
            list_of_docids = doc_ids
        results = dict()
        for docid in list_of_docids:
            if docid not in self._doc_embedding_dict:
                if print_warning:
                    print 'Warning. Your docid ' + docid + ' is not in the embeddings dictionary. Skipping...'
                continue
            scored_dict = self._generate_scored_dict(docid)
            results[docid] = WordEmbedding.extract_top_k(scored_dict, k=k)

        return results

    def get_vector(self, doc_ids, print_warning=True):
        """

        :param doc_ids: either a single doc_ids or a list.
        :param print_warning: if True (by default), it will print out a warning if it does not find a docid
        in the embeddings dictionary. Disable at your own risk.
        :return: a single vector (or None).
        If there were multiple doc_ids, the vectors will be added. To normalize the vector use VectorUtils. You can
        modify the returned vector in any way without affecting the 'original' vectors in the embeddings dict.
        """
        if type(doc_ids) != list:
            if doc_ids in self._doc_embedding_dict:
                return self._doc_embedding_dict[doc_ids]
            else:
                if print_warning:
                    print 'Warning. Your doc '+doc_ids+' is not in the embeddings dictionary. Returning None...'
                return None

        result = list()
        # We have a list of words
        for doc in doc_ids:
            if doc not in self._doc_embedding_dict:
                if print_warning:
                    print 'Warning. Your docid '+doc+' is not in the embeddings dictionary. Moving on to next doc...'
                continue
            result.append(list(self._doc_embedding_dict[doc]))
        if not result:
            if print_warning:
                print 'None of your docids were in the embeddings dictionary. Returning None...'
            return None
        else:
            return add_vectors(result)

    def _generate_scored_dict(self, doc_id):
        scored_dict = dict()
        seed_vector = self._doc_embedding_dict[doc_id]
        for token, vector in self._doc_embedding_dict.items():
            if token == doc_id:
                continue
            else:
                score = WordEmbedding.compute_abs_cosine_sim(seed_vector, vector)
                if score not in scored_dict:
                    scored_dict[score] = list()
                scored_dict[score].append(token)
        return scored_dict