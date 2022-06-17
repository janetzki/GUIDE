import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class DictionaryCreator(object):
    def __init__(self):
        self.source_language = 'eng-web'
        self.target_language = ''
        print(f'source language: {self.source_language}, target language: {self.target_language}')

        bibles_by_language = {
            'eng-kjv': 'eng-eng-kjv.txt',
            'eng-web': 'eng-eng-web.txt',
            'fra-fob': 'fra-fra_fob.txt',
            'fra-lsg': 'fra-fraLSG.txt',
            'spa': 'spa-spaRV1909.txt',
            'ind': 'ind-ind.txt',
            'deu': 'no semdoms available/deu-deuelo.txt',
            'rus': 'no semdoms available/rus-russyn.txt',
            # 'tha': '',
            # 'tel': '',
            # 'urd': '',
            # 'hin': '',
            # 'khm': '',
            # 'nep': '',
        }

        self.source_bible = bibles_by_language[self.source_language]
        self.target_bible = bibles_by_language[self.target_language]

        self.file_suffix = f'{self.source_language}-{self.target_language}'
        self.state_file_name = f'dc_state-{self.file_suffix}.pkl'
        self.base_path = '../experiments'
        self.data_path = os.path.join(self.base_path, 'data')
        # self.vectorizer = TfidfVectorizer(max_df=0.05, min_df=5) # without alignment
        self.vectorizer = TfidfVectorizer() #max_df=0.5, min_df=2)  # with alignment

        # Saved data
        self.source_verse_tokens_set = None
        self.source_qids_by_word = None
        self.source_question_by_qid = None
        self.target_verse_tokens_set = None
        self.gt_target_words_by_qid = None
        self.gt_target_qids_by_word = None
        self.aligned_target_words_by_qid = None
        self.top_tfidfs_by_qid = None

    def _save_state(self):
        with open(os.path.join(self.data_path, self.state_file_name), 'wb') as state_file:
            pickle.dump((self.source_verse_tokens_set,
                         self.source_qids_by_word,
                         self.source_question_by_qid,
                         self.target_verse_tokens_set,
                         self.gt_target_words_by_qid,
                         self.gt_target_qids_by_word,
                         self.aligned_target_words_by_qid,
                         self.top_tfidfs_by_qid),
                        state_file)

    def _load_state(self):
        with open(os.path.join(self.data_path, self.state_file_name), 'rb') as state_file:
            (self.source_verse_tokens_set,
             self.source_qids_by_word,
             self.source_question_by_qid,
             self.target_verse_tokens_set,
             self.gt_target_words_by_qid,
             self.gt_target_qids_by_word,
             self.aligned_target_words_by_qid,
             self.top_tfidfs_by_qid) = pickle.load(state_file)

    def dc_preprocessing(self, save=False):
        def load_data():
            # load sds
            df_source = pd.read_csv(
                f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.source_language[:3]}.csv')

            if os.path.isfile(
                    f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.target_language[:3]}.csv'):
                df_target = pd.read_csv(
                    f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.target_language[:3]}.csv')
            else:
                print(
                    f'WARNING: unable to load {self.base_path}/../semdom extractor/output/semdom_qa_clean_{self.target_language[:3]}.csv')
                df_target = None
                assert (self.target_language == 'deu')

            # dfc = df.groupby(['cid', 'category']).agg(list)

            # load source and target bible
            scripture_dir = '../load bibles in DGraph/content/scripture_public_domain'
            source_verses = None
            target_verses = None
            with open(os.path.join(scripture_dir, self.source_bible), 'r') as eng:
                with open(os.path.join(scripture_dir, self.target_bible), 'r') as deu:
                    source_verses = eng.readlines()
                    target_verses = deu.readlines()
            assert (len(source_verses) == len(target_verses))
            return df_source, df_target, source_verses, target_verses

        df_source, df_target, source_verses, target_verses = load_data()

        # optional: increase performance by querying words from words_eng
        def build_sds(df):
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

        self.source_qids_by_word, _, self.source_question_by_qid = build_sds(df_source)
        if self.target_language != 'deu':
            self.gt_target_qids_by_word, self.gt_target_words_by_qid, _ = build_sds(df_target)

        def tokenize_verses(verses, file):
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.train(
                files=[file],
                trainer=trainer)
            tokens_by_verse = [tokenizer.encode(verse).tokens for verse in verses]
            return tokens_by_verse

        source_tokens_by_verse = tokenize_verses(source_verses, os.path.join(
            '../load bibles in DGraph/content/scripture_public_domain', self.source_bible))
        target_tokens_by_verse = tokenize_verses(target_verses, os.path.join(
            '../load bibles in DGraph/content/scripture_public_domain', self.target_bible))
        self.source_verse_tokens_set = set(
            [token.lower() for source_tokens in source_tokens_by_verse for token in source_tokens])
        self.target_verse_tokens_set = set(
            [token.lower() for target_tokens in target_tokens_by_verse for token in target_tokens])

        def combine_alignments():
            # combine source and target verses into a single file for word aligner
            with open(f'{self.data_path}/{self.file_suffix}.txt', 'w') as combined_bibles:
                for idx, (source_tokens, target_tokens) in enumerate(
                        zip(source_tokens_by_verse, target_tokens_by_verse)):
                    if len(source_tokens) * len(target_tokens) == 0 and len(source_tokens) + len(target_tokens) > 0:
                        # print(idx)  # verse is missing in one language
                        pass
                    if len(source_tokens) * len(target_tokens) == 0:
                        source_tokens = ['#placeholder#']
                        target_tokens = ['#placeholder#']
                    combined_bibles.write(' '.join(source_tokens) + ' ||| ' + ' '.join(target_tokens) + '\n')
            # print(subprocess.call(['sh', f"align_bibles.sh", self.source_language, self.target_language]))

        combine_alignments()

        """ maps words in target language to semantic domains """

        def map_target_words_to_qids():
            target_words_by_qid = defaultdict(str)

            with open(f'{self.data_path}/diag-{self.file_suffix}.align', 'r') as alignment_file:
                alignment = alignment_file.readlines()
                assert (len(alignment) == len(source_verses))
                for (idx, alignment_line), source_tokens, target_tokens in tqdm(
                        zip(enumerate(alignment), source_tokens_by_verse, target_tokens_by_verse),
                        desc=f'matching {self.target_language} words with semantic domain questions',
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
            return target_words_by_qid

        self.aligned_target_words_by_qid = map_target_words_to_qids()

        def show_mapped_words():
            idx = 1  # idx <= 4505 # 7938 questions, 1915 sds (for eng-deu)
            question = list(self.aligned_target_words_by_qid.keys())[idx]
            print(question)
            print(list((self.aligned_target_words_by_qid.values()))[idx])

        # show_mapped_words()

        if save:
            self._save_state()

    def dc_train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()

        def compute_tfidfs():
            tfidfs = self.vectorizer.fit_transform(list(self.aligned_target_words_by_qid.values()))
            # print(len(self.vectorizer.stop_words_))
            # print(self.vectorizer.get_feature_names_out())
            # print(tfidfs.shape)
            return tfidfs

        tfidfs = compute_tfidfs()

        def build_top_tfidfs():
            top_tfidfs_by_qid = {}
            assert(len(tfidfs) == len(self.aligned_target_words_by_qid))
            for idx, tfidf in tqdm(enumerate(tfidfs), desc='collecting top tf-idf scores', total=tfidfs.shape[0]):
                qid = list(self.aligned_target_words_by_qid.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=["TF-IDF"])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                top_tfidfs_by_qid[qid] = df.head(20)
            return top_tfidfs_by_qid

        self.top_tfidfs_by_qid = build_top_tfidfs()

        sorted(self.top_tfidfs_by_qid.items())[0]

        if save:
            self._save_state()

    def dc_evaluate(self, load=False):
        """ load source and corresponding target words from Purdue Team (ground truth data for dictionary creation) """

        if load:
            self._load_state()

        """ remove all target words with a TF-IDF value below a threshold """

        def filter_target_sds_with_threshold():
            threshold = 0.02
            filtered_target_sds = dict()
            for qid, tf_idfs_df in self.top_tfidfs_by_qid.items():
                filtered_target_sds[qid] = list(tf_idfs_df[tf_idfs_df['TF-IDF'] > threshold].index.values)
            return filtered_target_sds

        predicted_target_words_by_qid = filter_target_sds_with_threshold()

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

            # # How many non-unique words are in the ground-truth target semantic domains?
            # num_total_sd_source_words = 0
            # num_total_sd_words_in_source_verses = 0
            # for _, words_for_question in tqdm(self..items(),
            #                                   desc=f'collecting words in {self.source_language} semantic domains',
            #                                   total=len(self.)):
            #     num_total_sd_source_words += len(words_for_question)
            #     overlap = words_for_question & self.source_verse_tokens_set
            #     num_total_sd_words_in_source_verses += len(words_for_question & self.source_verse_tokens_set)

            # How many non-unique words are in the ground-truth target semantic domains?
            num_total_gt_target_words = 0
            num_total_gt_sd_words_in_target_verses = 0
            # num_total_single_gt_sd_words_in_target_verses = 0
            for _, words_for_question in tqdm(self.gt_target_words_by_qid.items(),
                                              desc=f'collecting words in {self.target_language} semantic domains',
                                              total=len(self.gt_target_words_by_qid)):
                num_total_gt_target_words += len(words_for_question)
                overlap = [word for word in words_for_question if word in self.target_verse_tokens_set]
                num_total_gt_sd_words_in_target_verses += len(overlap)
                # single_words = [word for word in overlap if ' ' not in word]
                # num_total_single_gt_sd_words_in_target_verses += len(single_words)

            # How many of the found target words actually appear in the ground-truth set?
            precision = num_true_positive_words / num_positive_words
            print(f'precision: {precision:.3f} ({num_true_positive_words} '
                  f'out of {num_positive_words} found {self.target_language} semantic domain words are correct)')

            # How many of the target sd words in the ground-truth set were actually found?
            recall = num_true_positive_words / num_total_gt_target_words
            print(f'recall: {recall:.3f} ({num_true_positive_words} '
                  f'out of {num_total_gt_target_words} {self.target_language} actual semantic domain words found)')

            f1 = 2 * (precision * recall) / (precision + recall)
            print(f'F1: {f1:.3f}\n')

            # # How many of the source words appear in the source verses?
            # source_word_coverage = num_total_sd_words_in_source_verses / num_total_sd_source_words
            # print(f'Source word coverage: {source_word_coverage:.3f} ({num_total_sd_words_in_source_verses} '
            #       f'out of {len(num_total_sd_source_words)} {self.source_language} actual non-unique semantic domain words '
            #       f'also appear in the source verses)')

            # How many of the gt target words appear in the target verses?
            target_word_coverage = num_total_gt_sd_words_in_target_verses / num_total_gt_target_words
            print(
                f'Ground truth target word coverage: {target_word_coverage:.3f} ({num_total_gt_sd_words_in_target_verses} '
                f'out of {num_total_gt_target_words} {self.target_language} actual non-unique semantic domain words '
                f'also appear in the target verses)')

            # How many of the target sd words in the ground-truth set - that also appear in the target verses -
            # was actually found?
            recall_adjusted = num_true_positive_words / num_total_gt_sd_words_in_target_verses
            print(f'recall*: {recall_adjusted:.3f} ({num_true_positive_words} '
                  f'out of {num_total_gt_sd_words_in_target_verses} {self.target_language} actual semantic domain words '
                  f'- that also appear in the target verses - found)')

            f1_adjusted = 2 * (precision * recall_adjusted) / (precision + recall_adjusted)
            print(f'F1*: {f1_adjusted:.3f}\n')

            # optional: consider word groups vs. single words in calculation
            # # How many of the single gt target words appear in the target verses?
            # target_word_coverage = num_total_single_gt_sd_words_in_target_verses / num_total_gt_target_words
            # print(f'Ground truth single target word coverage: {target_word_coverage:.3f} ({num_total_single_gt_sd_words_in_target_verses} '
            #       f'out of {num_total_gt_target_words} {self.target_language} actual non-unique semantic domain words '
            #       f'also appear in the target verses)')
            #
            # # How many of the target sd words in the ground-truth set - that also appear in the target verses -
            # # was actually found?
            # recall_adjusted2 = num_true_positive_words / num_total_single_gt_sd_words_in_target_verses
            # print(f'recall**: {recall_adjusted2:.3f} ({num_true_positive_words} '
            #       f'out of {num_total_single_gt_sd_words_in_target_verses} {self.target_language} actual single semantic domain words '
            #       f'- that also appear in the target verses - found)')
            #
            # f1_adjusted2 = 2 * (precision * recall_adjusted2) / (precision + recall_adjusted2)
            # print(f'F1**: {f1_adjusted2:.3f}')

            return f1

        compute_f1_score()

        def load_test_data():
            df_test = pd.read_csv(f'{self.data_path}/multilingual_semdom_dictionary.csv')
            df_test = df_test[[f'{self.source_language[:3]}-000.txt', f'{self.target_language[:3]}-000.txt']]
            df_test.columns = ['source_word', 'target_words']
            df_test = df_test[df_test['target_words'].notna()]

            df_test['source_word'] = df_test['source_word'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
            df_test['target_words'] = df_test['target_words'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
            df_test = df_test.explode('source_word').reset_index(drop=True)
            # df_test = df_test.groupby(['target_words']).agg(list)
            return df_test

        """ compute MRR to evaluate DC """

        def compute_mean_reciprocal_rank():
            # Filter target question that we are going to check because ground truth set is limited:
            # We only consider questions which have at least one source word in the gt set with a target translation.
            if self.target_language == 'urd':
                return None
            target_qids = defaultdict(list)
            df_test = load_test_data()
            for source_word, qids in tqdm(self.source_qids_by_word.items(),
                                          desc=f'filtering {self.target_language} question ids',
                                          total=len(self.source_qids_by_word)):
                target_words = list(df_test.query(f'source_word=="{source_word}"')['target_words'])
                if len(target_words) == 0:
                    continue
                target_words = target_words[0]
                for qid in qids:
                    if qid in self.top_tfidfs_by_qid:
                        target_qids[qid].extend(target_words)
                    # some semantic domains are missing in the target sds because no aligned words were found

            # in all selected target top_tfidfs, look for first ranked target word that also appears in df_test (gt data)
            mean_reciprocal_rank = 0
            for qid, target_words in target_qids.items():
                word_list = list(self.top_tfidfs_by_qid[qid].index)
                reciprocal_rank = 0
                for idx, word in enumerate(word_list):
                    if word in target_words:
                        reciprocal_rank = 1 / (idx + 1)
                        break
                print(qid)
                print('\t', self.source_question_by_qid[qid])
                print('\t found words:', word_list)
                print('\t reference words:', target_words)
                print(f'\t RR: {reciprocal_rank:.2f}\n')
                mean_reciprocal_rank += reciprocal_rank
            mean_reciprocal_rank /= len(target_qids)
            print(
                f"{len(target_qids)} of {len(self.top_tfidfs_by_qid)} {self.target_language} questions selected")
            return mean_reciprocal_rank

        print(f'MRR: {compute_mean_reciprocal_rank():.3f}')


if __name__ == '__main__':
    dc = DictionaryCreator()
    dc.dc_preprocessing(save=True)
    dc.dc_train_tfidf_based_model(load=True, save=True)
    dc.dc_evaluate(load=True)
