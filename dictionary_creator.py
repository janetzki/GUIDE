import os
from collections import defaultdict

import dill
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


class DictionaryCreator(object):
    def __init__(self):
        self.source_languages = ['eng-kjv', 'eng-web']
        self.target_languages = ['deu', 'rus']  # 'fra-fob', 'fra-lsg', 'spa', 'ind', 'tel', 'tha', 'hin', 'nep', 'urd',
        self.languages = self.source_languages + self.target_languages
        print(f'source languages: {self.source_languages}, target languages: {self.target_languages}')

        self.bibles_by_lang = {
            'eng-kjv': 'eng-eng-kjv.txt',
            'eng-web': 'eng-eng-web.txt',
            'fra-fob': 'fra-fra_fob.txt',
            'fra-lsg': 'fra-fraLSG.txt',
            'spa': 'spa-spaRV1909.txt',
            'ind': 'ind-ind.txt',
            'deu': 'no semdoms available/deu-deuelo.txt',
            'rus': 'no semdoms available/rus-russyn.txt',
            'tha': 'tha-thaKJV.txt',
            'tel': 'tel-telirv.txt',
            'urd': 'urd-urdgvu.txt',
            'hin': 'hin-hinirv.txt',
            'nep': 'nep-nepulb.txt',
        }

        self.state_file_name = f'dc_state-{self.source_languages}-{self.target_languages}.dill'
        self.base_path = '../experiments'
        self.data_path = os.path.join(self.base_path, 'data')
        self.vectorizer = TfidfVectorizer()

        # Saved data
        self.sds_by_lang = None
        self.verses_by_lang = None
        self.qids_by_word_by_lang = None
        self.words_by_qid_by_lang = None
        self.question_by_qid_by_lang = None
        self.tokens_by_verse_by_lang = None
        self.token_set_by_lang = None
        self.aligned_target_words_by_qid_by_lang = None
        self.top_tfidfs_by_qid_by_lang = None
        self.top_qids_by_word_by_lang = None

    def _save_state(self):
        # save all class variables to a dill file
        with open(os.path.join(self.data_path, self.state_file_name), 'wb') as state_file:
            dill.dump((self.sds_by_lang,
                       self.verses_by_lang,
                       self.qids_by_word_by_lang,
                       self.words_by_qid_by_lang,
                       self.question_by_qid_by_lang,
                       self.tokens_by_verse_by_lang,
                       self.token_set_by_lang,
                       self.aligned_target_words_by_qid_by_lang,
                       self.top_tfidfs_by_qid_by_lang,
                       self.top_qids_by_word_by_lang),
                      state_file)

    def _load_state(self):
        with open(os.path.join(self.data_path, self.state_file_name), 'rb') as state_file:
            (self.sds_by_lang,
             self.verses_by_lang,
             self.qids_by_word_by_lang,
             self.words_by_qid_by_lang,
             self.question_by_qid_by_lang,
             self.tokens_by_verse_by_lang,
             self.token_set_by_lang,
             self.aligned_target_words_by_qid_by_lang,
             self.top_tfidfs_by_qid_by_lang,
             self.top_qids_by_word_by_lang) = dill.load(state_file)

    def _load_data(self):
        # load sds and bible verses for all languages
        self.sds_by_lang = {}
        self.verses_by_lang = {}

        for lang in self.languages:
            # load sds
            sd_path = f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{lang[:3]}.csv'
            if os.path.isfile(sd_path):
                self.sds_by_lang[lang] = pd.read_csv(sd_path)
            else:
                print(f'WARNING: unable to load {sd_path}')
                # create empty dataframe
                self.sds_by_lang[lang] = pd.DataFrame(
                    {'cid': [], 'category': [], 'question_index': [], 'question': [], 'answer': []})
                assert (lang in ('deu', 'rus'))

            # load bible verses
            with open(
                    os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_lang[lang]),
                    'r') as bible:
                self.verses_by_lang[lang] = bible.readlines()
            assert (len(self.verses_by_lang[lang]) == 41899)

    def _build_sds(self):
        # convert sd dataframe to dictionary
        # optional: increase performance by querying words from words_eng
        self.qids_by_word_by_lang = defaultdict(lambda: defaultdict(set))
        self.words_by_qid_by_lang = defaultdict(dict)
        self.question_by_qid_by_lang = defaultdict(dict)

        for lang, sds in self.sds_by_lang.items():
            for index, row in sds.iterrows():
                question = row.question.replace("'", '')
                question = question.replace('"', '')
                answer = row.answer.replace("'", '')
                answer = answer.replace('"', '')
                words = [word.strip() for word in answer.split(',') if word]
                qid = f"{row.cid} {row.question_index}"
                self.words_by_qid_by_lang[lang][qid] = words
                self.question_by_qid_by_lang[lang][qid] = question
                for word in words:
                    self.qids_by_word_by_lang[lang][word].add(qid)

    def _tokenize_verses(self):
        self.tokens_by_verse_by_lang = {}
        self.token_set_by_lang = {}

        for lang in self.languages:
            file = os.path.join(
                '../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_lang[lang])
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.train(
                files=[file],
                trainer=trainer)
            tokens_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_lang[lang]]
            self.tokens_by_verse_by_lang[lang] = tokens_by_verse
            self.token_set_by_lang[lang] = set([token.lower() for tokens in tokens_by_verse for token in tokens])

    def _combine_alignments(self):
        # combine source and target verses into a single file for word aligner
        for lang_1 in self.languages:
            for lang_2 in self.languages:
                if lang_1 >= lang_2:
                    continue
                with open(f'{self.data_path}/{lang_1}-{lang_2}.txt', 'w') as combined_bibles:
                    for idx, (lang_1_tokens, lang_2_tokens) in enumerate(
                            zip(self.tokens_by_verse_by_lang[lang_1], self.tokens_by_verse_by_lang[lang_2])):
                        if len(lang_1_tokens) * len(lang_2_tokens) == 0 and len(lang_1_tokens) + len(lang_2_tokens) > 0:
                            # print(idx)  # verse is missing in one language
                            pass
                        if len(lang_1_tokens) * len(lang_2_tokens) == 0:
                            lang_1_tokens = ['#placeholder#']
                            lang_2_tokens = ['#placeholder#']
                        combined_bibles.write(' '.join(lang_1_tokens) + ' ||| ' + ' '.join(lang_2_tokens) + '\n')
                # print(subprocess.call(['sh', f"align_bibles.sh", self.source_language, self.target_language]))

    def _map_target_words_to_qids(self):
        # map words in all target languages to semantic domains
        self.aligned_target_words_by_qid_by_lang = defaultdict(lambda: defaultdict(str))

        source_lang = self.source_languages[0]
        for target_lang in self.target_languages:
            with open(f'{self.data_path}/diag-{source_lang}-{target_lang}.align', 'r') as alignment_file:
                alignment = alignment_file.readlines()
                assert (len(alignment) == len(self.verses_by_lang[source_lang]))
                for (idx, alignment_line), source_tokens, target_tokens in tqdm(
                        zip(enumerate(alignment), self.tokens_by_verse_by_lang[source_lang],
                            self.tokens_by_verse_by_lang[target_lang]),
                        desc=f'matching {target_lang} words with {source_lang} semantic domain questions',
                        total=len(self.verses_by_lang[source_lang])):
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

                        if source_token not in self.qids_by_word_by_lang[source_lang]:
                            continue
                        new_qids = self.qids_by_word_by_lang[source_lang][source_token]
                        if len(new_qids) == 0:
                            continue
                        for new_qid in new_qids:
                            self.aligned_target_words_by_qid_by_lang[target_lang][new_qid] += ', ' + target_token

    def preprocess_data(self, save=False):
        self._load_data()
        self._build_sds()
        self._tokenize_verses()
        self._combine_alignments()
        self._map_target_words_to_qids()
        if save:
            self._save_state()

    def _build_top_tfidfs(self):
        self.top_tfidfs_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_languages:
            tfidfs = self.vectorizer.fit_transform(list(self.aligned_target_words_by_qid_by_lang[target_lang].values()))
            assert (tfidfs.shape[0] == len(self.aligned_target_words_by_qid_by_lang[target_lang]))
            for idx, tfidf in tqdm(enumerate(tfidfs), desc='collecting top tf-idf scores', total=tfidfs.shape[0]):
                qid = list(self.aligned_target_words_by_qid_by_lang[target_lang].keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=["TF-IDF"])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                self.top_tfidfs_by_qid_by_lang[target_lang][qid] = df.head(20)

    def train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()
        self._build_top_tfidfs()

        # build self.top_qids_by_word_by_lang
        self.top_qids_by_word_by_lang = defaultdict(lambda: defaultdict(list))
        for target_lang in self.target_languages:
            for qid, tfidfs_df in self.top_tfidfs_by_qid_by_lang[target_lang].items():
                for word, tfidf in zip(list(tfidfs_df.index.values), list(tfidfs_df['TF-IDF'])):
                    self.top_qids_by_word_by_lang[target_lang][word].append((qid, tfidf))

        if save:
            self._save_state()

    def _filter_target_sds_with_threshold(self):
        # remove all target words with a TF-IDF value below a threshold
        threshold = 0.15
        filtered_target_words_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_languages:
            for qid, tfidfs_df in self.top_tfidfs_by_qid_by_lang[target_lang].items():
                filtered_target_words_by_qid_by_lang[target_lang][qid] = list(
                    tfidfs_df[tfidfs_df['TF-IDF'] > threshold].index.values)
        return filtered_target_words_by_qid_by_lang

    def _compute_f1_score(self, predicted_target_words_by_qid, target_lang):
        """
        Compute precision, recall, and F1 score to evaluate DC. This requires a ground-truth semantic domain
        dictionary for the target language.
        """
        num_positive_words = 0
        num_true_positive_words = 0
        gt_target_words_by_qid = self.words_by_qid_by_lang[target_lang]
        for qid, words in tqdm(predicted_target_words_by_qid.items(),
                               desc=f'counting true positive words in {target_lang} semantic domains',
                               total=len(predicted_target_words_by_qid)):
            num_positive_words += len(words)
            for word in words:
                num_true_positive_words += word in gt_target_words_by_qid.get(qid, [])

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
        for _, words_for_question in tqdm(gt_target_words_by_qid.items(),
                                          desc=f'collecting words in {target_lang} semantic domains',
                                          total=len(gt_target_words_by_qid)):
            num_total_gt_target_words += len(words_for_question)
            overlap = [word for word in words_for_question if word in self.token_set_by_lang[target_lang]]
            num_total_gt_sd_words_in_target_verses += len(overlap)
            # single_words = [word for word in overlap if ' ' not in word]
            # num_total_single_gt_sd_words_in_target_verses += len(single_words)

        if num_total_gt_target_words == 0:
            print(
                f'Cannot compute F1 score etc. for {target_lang} because no ground-truth target semantic domains have been loaded')
            return

        # How many of the found target words actually appear in the ground-truth set?
        precision = num_true_positive_words / num_positive_words
        print(f'precision: {precision:.3f} ({num_true_positive_words} '
              f'out of {num_positive_words} found {target_lang} semantic domain words are correct)')

        # How many of the target sd words in the ground-truth set were actually found?
        recall = num_true_positive_words / num_total_gt_target_words
        print(f'recall: {recall:.3f} ({num_true_positive_words} '
              f'out of {num_total_gt_target_words} {target_lang} actual semantic domain words found)')

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
            f'out of {num_total_gt_target_words} {self.target_languages} actual non-unique semantic domain words '
            f'also appear in the target verses)')

        # How many of the target sd words in the ground-truth set - that also appear in the target verses -
        # was actually found?
        recall_adjusted = num_true_positive_words / num_total_gt_sd_words_in_target_verses
        print(f'recall*: {recall_adjusted:.3f} ({num_true_positive_words} '
              f'out of {num_total_gt_sd_words_in_target_verses} {self.target_languages} actual semantic domain words '
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

    def _load_test_data(self, source_lang, target_lang):
        # load source and corresponding target words from Purdue Team (ground truth data for dictionary creation)
        df_test = pd.read_csv(f'{self.data_path}/multilingual_semdom_dictionary.csv')
        df_test = df_test[[f'{source_lang[:3]}-000.txt', f'{target_lang[:3]}-000.txt']]
        df_test.columns = ['source_word', 'target_words']
        df_test = df_test[df_test['target_words'].notna()]

        df_test['source_word'] = df_test['source_word'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test['target_words'] = df_test['target_words'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test = df_test.explode('source_word').reset_index(drop=True)
        # df_test = df_test.groupby(['target_words']).agg(list)
        return df_test

    def _compute_mean_reciprocal_rank(self, target_lang):
        # compute MRR to evaluate DC
        # Filter target question that we are going to check because ground truth set is limited:
        # We only consider questions which have at least one source word in the gt set with a target translation.
        # TODO: Also filter out source words (and questions, if empty) that do not appear in the source verses. (e.g., "snake" does not appear in the KJV bible)
        if target_lang == 'urd':
            return None
        target_qids = defaultdict(list)
        source_lang = self.source_languages[0]
        df_test = self._load_test_data(source_lang, target_lang)
        for source_word, qids in tqdm(self.qids_by_word_by_lang[source_lang].items(),
                                      desc=f'filtering {target_lang} question ids',
                                      total=len(self.qids_by_word_by_lang[source_lang])):
            target_words = list(df_test.query(f'source_word=="{source_word}"')['target_words'])
            if len(target_words) == 0:
                continue
            target_words = target_words[0]
            for qid in qids:
                if qid in self.top_tfidfs_by_qid_by_lang[target_lang]:
                    target_qids[qid].extend(target_words)
                # some semantic domains are missing in the target sds because no aligned words were found

        # in all selected target top_tfidfs, look for first ranked target word that also appears in df_test (gt data)
        mean_reciprocal_rank = 0
        for qid, target_words in target_qids.items():
            word_list = list(self.top_tfidfs_by_qid_by_lang[target_lang][qid].index)
            reciprocal_rank = 0
            for idx, word in enumerate(word_list):
                if word in target_words:
                    reciprocal_rank = 1 / (idx + 1)
                    break
            print(qid)
            print('\t', self.question_by_qid_by_lang[source_lang][qid])
            print('\t found (positive) words:', word_list)
            print('\t reference (true) words:', target_words)
            print('\t reciprocal rank:       ', f'{reciprocal_rank:.2f}\n')
            mean_reciprocal_rank += reciprocal_rank
        mean_reciprocal_rank /= len(target_qids)
        print(
            f"{len(target_qids)} of {len(self.top_tfidfs_by_qid_by_lang[target_lang])} {target_lang} questions selected")
        print(f'MRR: {mean_reciprocal_rank:.3f}')

    def evaluate(self, load=False):
        if load:
            self._load_state()
        filtered_target_words_by_qid_by_lang = self._filter_target_sds_with_threshold()
        for target_lang in self.target_languages:
            self._compute_f1_score(filtered_target_words_by_qid_by_lang[target_lang], target_lang)
            self._compute_mean_reciprocal_rank(target_lang)


if __name__ == '__main__':
    dc = DictionaryCreator()
    dc.preprocess_data(save=True)
    dc.train_tfidf_based_model(load=True, save=True)
    dc.evaluate(load=True)
