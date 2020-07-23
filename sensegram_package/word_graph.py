import faiss
import codecs
from time import time
from gensim.models import KeyedVectors


def compute_graph_of_related_words(vectors_fpath, neighbours_fpath, neighbors=200):
    print("Start collection of word neighbours.")
    tic = time()
    index, w2v = build_vector_index(vectors_fpath)
    print("Loaded word vectors")
    compute_neighbours(index, w2v, neighbours_fpath, neighbors)
    print("Elapsed: {:f} sec.".format(time() - tic))


def build_vector_index(w2v_fpath):
    print("Read w2v file")
    w2v = KeyedVectors.load_word2vec_format(w2v_fpath, binary=False, unicode_errors='ignore')
    print("Loaded w2V file")
    w2v.init_sims(replace=True)
    print("Init sims done")
    index = faiss.IndexFlatIP(w2v.vector_size)
    print("Got index from faiss model")
    index.add(w2v.syn0norm)

    return index, w2v


def compute_neighbours(index, w2v, nns_fpath, neighbors=200):
    print("Compute neighbours")
    tic = time()
    with codecs.open(nns_fpath, "w", "utf-8") as output:
        print("Get syn0norm from w2v")
        X = w2v.syn0norm
        
        print("get indices")
        D, I = index.search(X, neighbors + 1)

        j = 0
        num_iter = len(D)
        print_iter = num_iter/100
        print(f"Running for {num_iter} iterations")
        for _D, _I in zip(D, I):
            for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
                if n > 0:
                    output.write("{}\t{}\t{:f}\n".format(w2v.index2word[j], w2v.index2word[i], d))
            j += 1
            if j%print_iter == 0:
                print(f"Finished {j} iterations")

        print("Word graph:", nns_fpath)
        print("Elapsed: {:f} sec.".format(time() - tic))


