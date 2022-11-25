import pprint

from dictionary_creator import DictionaryCreator

pp = pprint.PrettyPrinter(indent=4)


def sdi_with_dictionary_lookup():
    dc = DictionaryCreator()

    # choose bible translations as evaluation data
    # tokenize the bible
    # dc.dc_preprocessing(save=True)
    # dc.dc__train_tfidf_based_model(load=True, save=True)
    dc._load_state()

    # lookup each token in the dictionary to indentify semantic domains
    scored_qids = list()
    for word in dc.target_tokens_by_verse[0]:
        normalized_word = word.lower()
        if normalized_word in dc.top_qids_by_word:
            for qid, tfidf in dc.top_qids_by_word[normalized_word.lower()]:
                question = dc.source_question_by_qid[qid]
                scored_qids.append((normalized_word, qid, question, tfidf))
    pp.pprint(scored_qids)

    # TODO: evaluate identified semantic domains with verse-semdom mappings from human labeler


def sdi_with_word_clustering():
    # Author: Olivier Grisel <olivier.grisel@ensta.org>
    #         Lars Buitinck
    #         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
    # License: BSD 3 clause

    from time import time
    import matplotlib.pyplot as plt
    import math

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation

    n_samples = 2000
    n_features = 10000
    n_components = 10
    n_top_words = 20

    def plot_top_words(model, feature_names, n_top_words, title):
        rows = math.ceil(n_components / 5)
        fig, axes = plt.subplots(rows, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            # print(top_features)
            # continue

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        # return
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()

    # Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
    # to filter out useless terms early on: the posts are stripped of headers,
    # footers and quoted replies, and common English words, words occurring in
    # only one document or in at least 95% of the documents are removed.

    # print("Loading dataset...")
    # t0 = time()
    # data, _ = fetch_20newsgroups(
    #    shuffle=True,
    #    random_state=1,
    #    remove=("headers", "footers", "quotes"),
    #    return_X_y=True,
    # )
    # data_samples = data[:n_samples]
    # print("done in %0.3fs." % (time() - t0))

    data_samples = []

    ## verse concatenation approach
    # step_size = 1
    # for idx in range(0, len(target_verses), step_size):
    #    data_samples.append(" ".join(target_verses[idx:idx+step_size]))

    ## sliding window approch
    # n_gram_size = 7
    # slide = 3
    # for verse in source_verses:
    # tokens = tokenizer_function(verse)
    # for idx in range(0, max(1, len(tokens) - n_gram_size + 1), slide):
    # data_samples.append(' '.join(tokens[idx:idx+n_gram_size]))

    data_samples = source_verses  # target_verses
    data_samples

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.005, min_df=2,  # max_features=n_features, stop_words="english"
    )
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=0.005, min_df=2,  # max_features=n_features, stop_words="english"
    )
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print()

    # Fit the NMF model
    print(
        "Fitting the NMF model (Frobenius norm) with tf-idf features, "
        "n_samples=%d and n_features=%d..." % (n_samples, n_features)
    )
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1, alpha=0.1, l1_ratio=0.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)"
    )

    # Fit the NMF model
    print(
        "\n" * 2,
        "Fitting the NMF model (generalized Kullback-Leibler "
        "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
        % (n_samples, n_features),
    )
    t0 = time()
    nmf = NMF(
        n_components=n_components,
        random_state=1,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha=0.1,
        l1_ratio=0.5,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf,
        tfidf_feature_names,
        n_top_words,
        "Topics in NMF model (generalized Kullback-Leibler divergence)",
    )

    print(
        "\n" * 2,
        "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
        % (n_samples, n_features),
    )
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    tf_feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")


if __name__ == '__main__':
    sdi_with_dictionary_lookup()
