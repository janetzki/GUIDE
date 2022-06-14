import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict


class DictionaryCreator(object):
    def __init__(self, source_language, target_language):
        print(f'source language: {source_language}, target language: {target_language}')
        self.source_language = source_language
        self.target_language = target_language
        self.base_path = '../experiments'
        self.data_path = os.path.join(self.base_path, 'data')
        # self.vectorizer = TfidfVectorizer(max_df=0.05, min_df=5) # without alignment
        self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=2)  # with alignment

        self.source_qids_by_word = None
        self.source_question_by_qid = None

        self.gt_target_words_by_qid = None
        self.aligned_target_words_by_qid = None
        self.target_qids_by_word = None

        self.top_tfidfs_by_qid = None

    def _save_state(self):
        with open(os.path.join(self.data_path, 'dc_state.pkl'), 'wb') as pklfile:
            pickle.dump((self.gt_target_words_by_qid,
                         self.source_qids_by_word,
                         self.aligned_target_words_by_qid,
                         self.target_qids_by_word,
                         self.top_tfidfs_by_qid),
                        pklfile)

    def _load_state(self):
        with open(os.path.join(self.data_path, 'dc_state.pkl'), 'rb') as pklfile:
            (self.source_qids_by_word,
             self.gt_target_words_by_qid,
             self.aligned_target_words_by_qid,
             self.target_qids_by_word,
             self.top_tfidfs_by_qid) = pickle.load(pklfile)

    def dc_preprocessing(self, save=False):
        source_bible = 'eng-eng-kjv.txt'
        target_bible = 'fra-fra_fob.txt'  # 'fra-fraLSG.txt' 'deu-deuelo.txt'

        def load_data():
            # load semdoms
            df_source = pd.read_csv(
                f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.source_language}.csv')
            df_target = pd.read_csv(
                f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.target_language}.csv')
            # dfc = df.groupby(['cid', 'category']).agg(list)

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

        # optional: increase performance by querying words from words_eng
        def build_semdoms(df):
            qid_by_word = defaultdict(set)
            words_by_qid = dict()
            question_by_qid = dict()

            for index, row in df.iterrows():
                question = row.question.replace("'", '')
                question = question.replace('"', '')
                answer = row.answer.replace("'", '')
                answer = answer.replace('"', '')
                words = [word.strip() for word in answer.split(',') if word]
                qid = f"{row.cid} {row.question_index}"
                words_by_qid[qid] = words
                question_by_qid[qid] = question
                for word in words:
                    qid_by_word[word].add(qid)

            return qid_by_word, words_by_qid, question_by_qid

        self.source_qids_by_word, _, self.source_question_by_qid = build_semdoms(df_source)
        self.target_qids_by_word, self.gt_target_words_by_qid, _ = build_semdoms(df_target)

        def tokenize_all_verses():
            # optional: Use tokenizer from huggingface
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
            target_words_by_qid = defaultdict(str)
            matched_qids = set()

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

                        if source_token not in self.source_qids_by_word:
                            continue
                        new_qids = self.source_qids_by_word[source_token]
                        if len(new_qids) == 0:
                            continue
                        for new_qid in new_qids:
                            target_words_by_qid[new_qid] += ', ' + target_token
                        matched_qids = {*matched_qids, *new_qids}
            return target_words_by_qid

        self.aligned_target_words_by_qid = map_target_words_to_semdoms()

        def show_mapped_words():
            idx = 1  # idx <= 4505 # 7938 questions, 1915 semdoms (for eng-deu)
            semdom_name = list(self.aligned_target_words_by_qid.keys())[idx]
            print(semdom_name)
            print(list((self.aligned_target_words_by_qid.values()))[idx])

        show_mapped_words()

        if save:
            self._save_state()

    def dc_train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()

        def compute_tfidfs():
            tfidfs = self.vectorizer.fit_transform(list(self.aligned_target_words_by_qid.values()))
            print(len(self.vectorizer.stop_words_))
            print(self.vectorizer.get_feature_names_out())
            print(tfidfs.shape)
            return tfidfs

        tfidfs = compute_tfidfs()

        def build_top_tfidfs():
            top_tfidfs_by_qid = {}
            for idx, tfidf in tqdm(enumerate(tfidfs), desc='collecting top tf-idf scores', total=tfidfs.shape[0]):
                qid = list(self.aligned_target_words_by_qid.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=["TF-IDF"])
                df = df.sort_values('TF-IDF', ascending=False)
                # TODO: cut off words with score 0.0
                top_tfidfs_by_qid[qid] = df.head(20)
            return top_tfidfs_by_qid

        self.top_tfidfs_by_qid = build_top_tfidfs()

        sorted(self.top_tfidfs_by_qid.items())[0]

        if save:
            self._save_state()

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
            self._load_state()

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
            for source_word, semdoms in tqdm(self.source_qids_by_word.items(),
                                             desc=f'filtering {self.target_language} semantic domains',
                                             total=len(self.source_qids_by_word)):
                target_words = list(df_test.query(f'source_word=="{source_word}"')['target_words'])
                if len(target_words) == 0:
                    continue
                target_words = target_words[0]
                for semdom in semdoms:
                    if semdom in self.top_tfidfs_by_qid:
                        target_semdoms[semdom].extend(target_words)
                    # some semdoms are missing in the target semdoms because no aligned words were found
            print(
                f"{len(target_semdoms)} of {len(self.top_tfidfs_by_qid)} {self.target_language} semdoms selected")

            # in all selected target top_tfidfs, look for first ranked target word that also appears in df_test (gt data)
            mean_reciprocal_rank = 0
            for semdom_question, target_words in target_semdoms.items():
                word_list = list(self.top_tfidfs_by_qid[semdom_question].index)
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

        # print(f'MRR: {compute_mean_reciprocal_rank()}')

        """ remove all target words with a TF-IDF value below a threshold """

        def filter_target_semdoms_with_threshold():
            threshold = 0.02
            filtered_target_semdoms = dict()
            for qid, tf_idfs_df in self.top_tfidfs_by_qid.items():
                filtered_target_semdoms[qid] = list(tf_idfs_df[tf_idfs_df['TF-IDF'] > threshold].index.values)
            return filtered_target_semdoms

        predicted_target_words_by_qid = filter_target_semdoms_with_threshold()

        """
        Compute precision, recall, and F1 score to evaluate DC. This requires a ground-truth semantic domain
        dictionary for the target language.
        """
        def compute_f1_score():
            num_positive_words = 0
            num_true_positive_words = 0
            for qid, words in tqdm(predicted_target_words_by_qid.items(),
                                   desc=f'counting true positive words in {self.target_language} semantic domains',
                                   total=len(predicted_target_words_by_qid)):
                num_positive_words += len(words)
                for word in words:
                    num_true_positive_words += word in self.gt_target_words_by_qid.get(qid, [])

            num_ground_truth_target_words = 0
            for _, semdoms in tqdm(self.target_qids_by_word.items(),
                                   desc=f'counting words in {self.target_language} semantic domains',
                                   total=len(self.target_qids_by_word)):
                num_ground_truth_target_words += len(semdoms)

            # Which share of the found target words actually appears in the ground-truth set?
            precision = num_true_positive_words / num_positive_words
            print(f'precision: {precision} ({num_true_positive_words} '
                  f'out of {num_positive_words} found {self.target_language} words are correct)')

            # Which share of the target semdoms in the ground-truth set was actually found?
            recall = num_true_positive_words / num_ground_truth_target_words
            print(f'recall: {recall} ({num_true_positive_words} '
                  f'out of {num_ground_truth_target_words} {self.target_language} words found)')

            f1 = 2 * (precision * recall) / (precision + recall)
            print(f'F1: {f1}')
            return f1
        compute_f1_score()


if __name__ == '__main__':
    dc = DictionaryCreator('eng', 'fra')
    # dc.dc_preprocessing(save=True)
    # dc.dc_train_tfidf_based_model(load=True, save=True)
    dc.dc_evaluate(load=True)
    # sid_with_word_clustering()
