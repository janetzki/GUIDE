import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict


class DictionaryCreator(object):
    def __init__(self):
        self.source_language = 'eng'
        self.target_language = 'fra'
        self.base_path = '../experiments'
        self.data_path = os.path.join(self.base_path, 'data')
        # self.vectorizer = TfidfVectorizer(max_df=0.05, min_df=5) # without alignment
        self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=2)  # with alignment

        self.source_semdoms_by_word = None
        self.source_question_by_cid = None

        self.target_semdom_words = None
        self.target_semdoms_by_question = None
        self.target_semdoms_by_word = None

        self.target_question_by_cid = None
        self.top_tfidfs_by_question = None

    def dc_preprocessing(self, save=False):
        source_bible = 'eng-eng-kjv.txt'
        target_bible = 'fra-fra_fob.txt'  # 'fra-fraLSG.txt' 'deu-deuelo.txt'

        def load_data():
            # load semdoms
            df_source = pd.read_csv(
                f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.source_language}.csv')
            df_target = pd.read_csv(
                f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.target_language}.csv')
            # dfc = df.groupby(['cids', 'category']).agg(list)

            # load source and target bible
            scripture_dir = '../load bibles in DGraph/content/scripture_public_domain'
            source_verses = None
            target_verses = None
            with open(os.path.join(scripture_dir, source_bible), 'r') as eng:
                with open(os.path.join(scripture_dir, target_bible), 'r') as deu:
                    source_verses = eng.readlines()
                    target_verses = deu.readlines()
            assert (len(source_verses) == len(target_verses))
            return df_source, df_target, source_verses, target_verses

        df_source, df_target, source_verses, target_verses = load_data()

        # optional TODO: increase performance by querying words from semdom_words_eng
        def build_semdoms(df):
            semdoms_by_word = defaultdict(set)
            question_by_cid = dict()
            semdoms_by_question = dict()

            for index, row in df.iterrows():
                text = row.question.replace("'", '')
                text = text.replace('"', '')
                answer = row.answer.replace("'", '')
                answer = answer.replace('"', '')
                words = [word.strip() for word in answer.split(',') if word]
                question_by_cid[row.cids] = text
                semdoms_by_question[text] = words
                for word in words:
                    semdoms_by_word[word].add(row.cids)

            return semdoms_by_word, question_by_cid, semdoms_by_question

        self.source_semdoms_by_word, self.source_question_by_cid, _ = build_semdoms(df_source)
        self.target_semdoms_by_word, self.target_question_by_cid, self.target_semdoms_by_question = build_semdoms(df_target)

        def tokenize_all_verses():
            # optional TODO: Use tokenizer from huggingface
            tokenizer_function = self.vectorizer.build_tokenizer()

            source_tokens_by_verse = [tokenizer_function(source_verse) for source_verse in source_verses]
            target_tokens_by_verse = [tokenizer_function(target_verse) for target_verse in target_verses]
            return source_tokens_by_verse, target_tokens_by_verse

        source_tokens_by_verse, target_tokens_by_verse = tokenize_all_verses()

        def combine_alignments():
            # combine source and target verses into a single file for word aligner
            with open(f'{self.data_path}/{self.source_language}-{self.target_language}.txt', 'w') as combined_bibles:
                for idx, (source_tokens, target_tokens) in enumerate(
                        zip(source_tokens_by_verse, target_tokens_by_verse)):
                    if len(source_tokens) * len(target_tokens) == 0 and len(source_tokens) + len(target_tokens) > 0:
                        print(idx)  # verse is missing in one language
                    if len(source_tokens) * len(target_tokens) == 0:
                        source_tokens = ['#placeholder#']
                        target_tokens = ['#placeholder#']
                    combined_bibles.write(' '.join(source_tokens) + ' ||| ' + ' '.join(target_tokens) + '\n')

        # combine_alignments()

        # !fast_align/build/fast_align -i data/eng-fra.txt -d -o -v > data/eng-fra-forward.align
        # !fast_align/build/fast_align -i data/eng-fra.txt -d -o -v -r > data/eng-fra-reverse.align
        # !fast_align/build/atools -i data/eng-fra-forward.align -j data/eng-fra-reverse.align -c grow-diag-final-and > data/eng-fra-diag.align

        """ maps words in target language to semantic domains """

        def map_target_words_to_semdoms():
            target_semdom_words = defaultdict(str)
            matched_semdoms = set()

            with open(f'{self.data_path}/{self.source_language}-{self.target_language}-diag.align',
                      'r') as alignment_file:
                alignment = alignment_file.readlines()
                for alignment_line, source_tokens, target_tokens in tqdm(
                        zip(alignment, source_tokens_by_verse, target_tokens_by_verse),
                        desc='matching semantic domains',
                        total=len(source_verses)):
                    if alignment_line == '\n':
                        continue
                    aligned_token_pairs = alignment_line.split(' ')
                    aligned_token_pairs[-1].replace('\n', '')

                    for aligned_token_pair in aligned_token_pairs:
                        source_token_idx, target_token_idx = [int(num) for num in aligned_token_pair.split('-')]
                        if source_token_idx >= len(source_tokens) or target_token_idx >= len(target_tokens):
                            # print('Skipped:', aligned_token_pair, source_tokens, source_token_idx, target_tokens,
                            #       target_token_idx)
                            continue
                        source_token = source_tokens[source_token_idx]
                        target_token = target_tokens[target_token_idx]

                        if source_token not in self.source_semdoms_by_word:
                            continue
                        new_semdoms = self.source_semdoms_by_word[source_token]
                        if len(new_semdoms) == 0:
                            continue
                        for new_semdom in new_semdoms:
                            target_semdom_words[new_semdom] += ', ' + target_token
                        matched_semdoms = {*matched_semdoms, *new_semdoms}
            return target_semdom_words

        self.target_semdom_words = map_target_words_to_semdoms()

        def show_mapped_words():
            idx = 1  # idx <= 4505 # 7938 questions, 1915 semdoms (for eng-deu)
            semdom_name = list(self.target_semdom_words.keys())[idx]
            print(semdom_name)
            print(list((self.target_semdom_words.values()))[idx])

        show_mapped_words()

        if save:
            with open(os.path.join(self.data_path, 'dc_preprocessing_output.pkl'), 'wb') as pklfile:
                pickle.dump((self.target_semdom_words, self.source_semdoms_by_word, self.source_question_by_cid,
                             self.target_semdoms_by_word,
                             self.target_question_by_cid), pklfile)

    def dc_train_tfidf_based_model(self, load=False, save=False):
        if load:
            with open(os.path.join(self.data_path, 'dc_preprocessing_output.pkl'), 'rb') as pklfile:
                (self.target_semdom_words, self.source_semdoms_by_word, self.source_question_by_cid,
                 self.target_semdoms_by_word,
                 self.target_question_by_cid) = pickle.load(pklfile)

        def compute_tfidfs():
            tfidfs = self.vectorizer.fit_transform(list(self.target_semdom_words.values()))
            print(len(self.vectorizer.stop_words_))
            print(self.vectorizer.get_feature_names_out())
            print(tfidfs.shape)
            return tfidfs

        tfidfs = compute_tfidfs()

        def build_top_tfidfs():
            top_tfidfs_by_question = {}
            for idx, tfidf in tqdm(enumerate(tfidfs), desc='collecting top tf-idf scores', total=tfidfs.shape[0]):
                question = list(self.target_semdom_words.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=["TF-IDF"])
                df = df.sort_values('TF-IDF', ascending=False)
                # TODO: cut off words with score 0.0
                top_tfidfs_by_question[question] = df.head(20)
            return top_tfidfs_by_question

        self.top_tfidfs_by_question = build_top_tfidfs()

        sorted(self.top_tfidfs_by_question.items())[0]

        if save:
            with open(os.path.join(self.data_path, 'dc_train_tfidf_based_model_output.pkl'), 'wb') as pklfile:
                pickle.dump(self.top_tfidfs_by_question, pklfile)

    def dc_evaluate(self, load=False):
        """ H2 EVALUATE
        de Melo: Ok, nun wäre es auch gut ein **"Experimental Setup"** zu entwickeln,
        mit dem man evaluieren/quantifizieren kann wie gut verschiedene Methoden funktionieren.
        Am besten in mehreren Sprachen.

        Das Setup muss nicht perfekt sein -
        es gibt offensichtlich Wörter,
        über die man sich streiten kann.

        Typischerweise hat man aber eine **vorgefertigte Liste von Wörtern,**
        die man am ehesten erwartet
        und kann z.B. mittels **Mean Reciprocal Rank**
        (z.B. jeweils mit dem höchsten Rank eines Wortes aus dem Ground Truth Set)
        dann ein Gesamtscore errechnen
        und so verschiedene Methoden vergleichen.

        Bsp.:

        1st tf-idf ranked word appears in ground truth set --> ReciprocalRank = 1.0
        2nd tf-idf ranked word appears in ground truth set --> RR = 0.5
        none of the tf-idf ranked words appears in ground truth set --> RR = 0.0
        """

        """ load source and corresponding target words from Purdue Team (ground truth data for dictionary creation) """

        if load:
            with open(os.path.join(self.data_path, 'dc_preprocessing_output.pkl'), 'rb') as pklfile:
                (self.target_semdom_words, self.source_semdoms_by_word, self.source_question_by_cid,
                 self.target_semdoms_by_word,
                 self.target_question_by_cid) = pickle.load(pklfile)
            with open(os.path.join(self.data_path, 'dc_train_tfidf_based_model_output.pkl'), 'rb') as pklfile:
                self.top_tfidfs_by_question = pickle.load(pklfile)

        def load_test_data():
            df_test = pd.read_csv(f'{self.data_path}/multilingual_semdom_dictionary.csv')
            df_test = df_test[[f'{self.source_language}-000.txt', f'{self.target_language}-000.txt']]
            df_test.rename(
                columns={f'{self.source_language}-000.txt': 'source_word',
                         f'{self.target_language}-000.txt': 'target_words'},
                inplace=True)
            df_test = df_test[df_test['target_words'].notna()]

            df_test['source_word'] = df_test['source_word'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
            df_test['target_words'] = df_test['target_words'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
            df_test = df_test.explode('source_word').reset_index(drop=True)
            # df_test = df_test.groupby(['target_words']).agg(list)
            return df_test

        df_test = load_test_data()
        df_test

        """ compute MRR to evaluate DC """

        def compute_mean_reciprocal_rank():
            # Filter target semantic domains that we are going to check because ground truth set is limited:
            # We only consider semdoms which have at least one source word in the gt set with a target translation.
            target_semdoms = defaultdict(list)
            for source_word, semdoms in tqdm(self.source_semdoms_by_word.items(),
                                             desc=f'filtering {self.target_language} semantic domains',
                                             total=len(self.source_semdoms_by_word)):
                target_words = list(df_test.query(f'source_word=="{source_word}"')['target_words'])
                if len(target_words) == 0:
                    continue
                target_words = target_words[0]
                for semdom in semdoms:
                    if semdom in self.top_tfidfs_by_question:
                        target_semdoms[semdom].extend(target_words)
                    # some semdoms are missing in the target semdoms because no aligned words were found
            print(
                f"{len(target_semdoms)} of {len(self.top_tfidfs_by_question)} {self.target_language} semdoms selected")

            # in all selected target top_tfidfs, look for first ranked target word that also appears in df_test (gt data)
            mean_reciprocal_rank = 0
            for semdom_question, target_words in target_semdoms.items():
                word_list = list(self.top_tfidfs_by_question[semdom_question].index)
                print(semdom_question, word_list, target_words)
                reciprocal_rank = 0
                for idx, word in enumerate(word_list):
                    if word in target_words:
                        reciprocal_rank = 1 / (idx + 1)
                        break
                print(reciprocal_rank)
                mean_reciprocal_rank += reciprocal_rank
            mean_reciprocal_rank /= len(target_semdoms)
            return mean_reciprocal_rank

        print(f'MRR: {compute_mean_reciprocal_rank()}')

        """ remove all target words with a TF-IDF value below a threshold """

        def filter_target_semdoms_with_threshold():
            threshold = 0.5
            filtered_target_semdoms = dict()
            for question, tf_idfs_df in self.top_tfidfs_by_question.items():
                filtered_target_semdoms[question] = list(tf_idfs_df[tf_idfs_df['TF-IDF'] > threshold].index.values)
            return filtered_target_semdoms

        predicted_target_semdoms_by_question = filter_target_semdoms_with_threshold()
        print(predicted_target_semdoms_by_question)

        """ Compute precision, recall, and F1 score to evaluate DC. This requires a ground-truth semantic domain dictionary for the target language. """

        def compute_precision():
            # Which share of the found target semdoms actually appears in the ground-truth set?
            num_true_positive_words = 0
            num_ground_truth_target_words = 0

            for cid, words in tqdm(predicted_target_semdoms_by_question.items(),
                                   desc=f'counting true positive words in {self.target_language} semantic domains',
                                   total=len(predicted_target_semdoms_by_question)):
                num_true_positive_words += self.semdoms_by_question[cid]
                # todo: continue here

            for _, semdoms in tqdm(self.target_semdoms_by_word.items(),
                                   desc=f'counting words in {self.target_language} semantic domains',
                                   total=len(self.target_semdoms_by_word)):
                num_ground_truth_target_words += len(semdoms)

            return num_true_positive_words / num_ground_truth_target_words

        def compute_recall():
            # Which share of the target semdoms in the ground-truth set was actually found?
            recall = 0
            return recall

        def compute_f1_score(precision, recall):
            return 2 * (precision * recall) / (precision + recall)

        precision = compute_precision()
        recall = compute_recall()
        print(f'precision: {compute_precision()}')
        print(f'recall: {compute_recall()}')
        print(f'F1: {compute_f1_score(precision, recall)}')


def sid_with_word_clustering():
    # Author: Olivier Grisel <olivier.grisel@ensta.org>
    #         Lars Buitinck
    #         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
    # License: BSD 3 clause

    from time import time
    import matplotlib.pyplot as plt
    import math

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.datasets import fetch_20newsgroups

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
    dc = DictionaryCreator()
    # dc.dc_preprocessing(save=True)
    # dc.dc_train_tfidf_based_model(load=True, save=True)
    dc.dc_evaluate(load=True)
    # sid_with_word_clustering()
