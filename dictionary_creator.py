import os
import subprocess
from collections import defaultdict, Counter

import dill
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


class DictionaryCreator(object):
    class Word:
        def __init__(self, text, lang, qids=None, appears_in_bible=False):
            self.text = text
            self.iso_language = lang
            self.aligned_words = Counter()
            self.qids = set() if qids is None else qids
            self.appears_in_bible = appears_in_bible

        def __str__(self):
            return f'{self.iso_language}: {self.text}'

        def add_aligned_word(self, word):
            self.aligned_words[str(word)] += 1

    def __init__(self):
        self.bids = ['bid-eng-kjv', 'bid-eng-web', 'bid-deu',
                     'bid-fra-fob']  # 'bid-fra-fob', 'bid-fra-lsg', 'bid-spa', 'bid-ind', 'bid-tel', 'bid-tha', 'bid-hin', 'bid-nep', 'bid-urd', 'bid-rus'
        self.target_langs = [self._convert_bid_to_lang(bid) for bid in self.bids]

        self.bibles_by_bid = {
            'bid-eng-kjv': 'eng-eng-kjv.txt',
            'bid-eng-web': 'eng-eng-web.txt',
            'bid-fra-fob': 'fra-fra_fob.txt',
            'bid-fra-lsg': 'fra-fraLSG.txt',
            'bid-spa': 'spa-spaRV1909.txt',
            'bid-ind': 'ind-ind.txt',
            'bid-deu': 'no semdoms available/deu-deuelo.txt',
            'bid-rus': 'no semdoms available/rus-russyn.txt',
            'bid-tha': 'tha-thaKJV.txt',
            'bid-tel': 'tel-telirv.txt',
            'bid-urd': 'urd-urdgvu.txt',
            'bid-hin': 'hin-hinirv.txt',
            'bid-nep': 'nep-nepulb.txt',
        }

        self.state_file_name = f'dc_state-{self.bids}.dill'
        self.base_path = '../experiments'
        self.data_path = os.path.join(self.base_path, 'data')
        self.vectorizer = TfidfVectorizer()
        self.wtxts_by_verse_by_bid = None

        # Saved data
        self.sds_by_lang = None
        self.verses_by_bid = None
        self.words_by_text_by_lang = None
        self.question_by_qid_by_lang = None
        self.aligned_wtxts_by_qid_by_lang_by_lang = None
        self.top_tfidfs_by_qid_by_lang = None
        self.top_qids_by_wtxt_by_lang = None

    @staticmethod
    def _convert_bid_to_lang(bid):
        # example: bid-eng-kjv -> eng
        return bid[4:7]

    @staticmethod
    def _group_words_by_qid(words):
        wtxts_by_qid = defaultdict(list)
        for word in words.values():
            for qid in word.qids:
                wtxts_by_qid[qid].append(word.text)
        return wtxts_by_qid

    def _save_state(self):
        # save all class variables to a dill file
        with open(os.path.join(self.data_path, self.state_file_name), 'wb') as state_file:
            dill.dump((self.sds_by_lang,
                       self.verses_by_bid,
                       self.words_by_text_by_lang,
                       self.question_by_qid_by_lang,
                       self.aligned_wtxts_by_qid_by_lang_by_lang,
                       self.top_tfidfs_by_qid_by_lang,
                       self.top_qids_by_wtxt_by_lang),
                      state_file)

    def _load_state(self):
        with open(os.path.join(self.data_path, self.state_file_name), 'rb') as state_file:
            (self.sds_by_lang,
             self.verses_by_bid,
             self.words_by_text_by_lang,
             self.question_by_qid_by_lang,
             self.aligned_wtxts_by_qid_by_lang_by_lang,
             self.top_tfidfs_by_qid_by_lang,
             self.top_qids_by_wtxt_by_lang) = dill.load(state_file)

    def _load_data(self):
        # load sds and bible verses for all languages
        self.sds_by_lang = {}
        self.verses_by_bid = {}

        languages = set([self._convert_bid_to_lang(bid) for bid in self.bids])
        for lang in languages:
            # load sds
            sd_path = f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{lang}.csv'
            if os.path.isfile(sd_path):
                self.sds_by_lang[lang] = pd.read_csv(sd_path)
            else:
                print(f'WARNING: unable to load {sd_path}')
                # create empty dataframe
                self.sds_by_lang[lang] = pd.DataFrame(
                    {'cid': [], 'category': [], 'question_index': [], 'question': [], 'answer': []})
                assert (lang in ('deu', 'rus'))

        for bid in self.bids:
            # load bible verses
            with open(
                    os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid]),
                    'r') as bible:
                self.verses_by_bid[bid] = bible.readlines()
            assert (len(self.verses_by_bid[bid]) == 41899)

    def _build_sds(self):
        # convert sd dataframe to dictionary
        # optional: increase performance by querying wtxts from words_eng
        # optional: increase performance by using dataframe instead of dict
        self.words_by_text_by_lang = defaultdict(dict)
        self.question_by_qid_by_lang = defaultdict(dict)

        for lang, sds in self.sds_by_lang.items():
            for index, row in sds.iterrows():
                question = row.question.replace("'", '')
                question = question.replace('"', '')
                answer = row.answer.replace("'", '')
                answer = answer.replace('"', '')
                qid = f'{row.cid} {row.question_index}'
                words = {wtxt: self.Word(wtxt.strip(), lang, [qid]) for wtxt in answer.split(',') if wtxt}

                # add new words to words_by_text_by_lang
                for word in words.values():
                    if word.text not in self.words_by_text_by_lang[lang]:
                        self.words_by_text_by_lang[lang][word.text] = word
                    else:
                        self.words_by_text_by_lang[lang][word.text].qids.append(qid)

                self.question_by_qid_by_lang[lang][qid] = question

    def _tokenize_verses(self):
        self.wtxts_by_verse_by_bid = {}

        for bid in self.bids:
            file = os.path.join(
                '../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid])
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer()  # todo: fix tokenizer (e.g., splits 'Prahlerei' into 'Pra' and 'hlerei') (might not be so important because this mainly happens for rare words)
            tokenizer.train(files=[file], trainer=trainer)

            # tokenize all verses
            wtxts_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_bid[bid]]
            # lowercase all wtxts
            wtxts_by_verse = [[wtxt.lower() for wtxt in verse] for verse in wtxts_by_verse]

            self.wtxts_by_verse_by_bid[bid] = wtxts_by_verse.copy()

            # mark words as appearing in the bible
            lang = self._convert_bid_to_lang(bid)
            wtxts_set = set([wtxt for wtxts in wtxts_by_verse for wtxt in wtxts])
            for wtxt in wtxts_set:
                if wtxt in self.words_by_text_by_lang[lang]:
                    self.words_by_text_by_lang[lang][wtxt].appears_in_bible = True
                else:
                    self.words_by_text_by_lang[lang][wtxt] = self.Word(wtxt, lang, [], True)

    def _combine_alignments(self):
        # combine verses from two different bibles into a single file for wtxt aligner
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2:
                    continue
                with open(f'{self.data_path}/{bid_1}-{bid_2}.txt', 'w') as combined_bibles:
                    for idx, (bid_1_wtxts, bid_2_wtxts) in enumerate(
                            zip(self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2])):
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0 and len(bid_1_wtxts) + len(bid_2_wtxts) > 0:
                            # print(idx)  # verse is missing in one bible
                            pass
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0:
                            bid_1_wtxts = ['#placeholder#']
                            bid_2_wtxts = ['#placeholder#']
                        combined_bibles.write(' '.join(bid_1_wtxts) + ' ||| ' + ' '.join(bid_2_wtxts) + '\n')
                if not os.path.isfile(f'{self.base_path}/data/diag_[{bid_1}]_[{bid_2}].align'):
                    print(subprocess.call(['sh', 'align_bibles.sh', bid_1, bid_2]))

    def _add_bidirectional_edge(self, wtxt1, wtxt2, lang1, lang2):
        # if wtxt1 not in self.words_by_text_by_lang[lang1]:
        #     self.words_by_text_by_lang[lang1][wtxt1] = self.Word(wtxt1, lang1)
        # if wtxt2 not in self.words_by_text_by_lang[lang2]:
        #     self.words_by_text_by_lang[lang2][wtxt2] = self.Word(wtxt2, lang2)
        word1 = self.words_by_text_by_lang[lang1][wtxt1]
        word2 = self.words_by_text_by_lang[lang2][wtxt2]
        word1.add_aligned_word(word2)
        word2.add_aligned_word(word1)

    def _map_two_bibles(self, alignment, bid_1, bid_2):
        # map words in two bibles to semantic domains
        # caveat: This function ignores wtxts that could not be aligned.
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)
        for alignment_line, wtxts_1, wtxts_2 in tqdm(
                zip(alignment, self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2]),
                desc=f'matching {bid_1} and {bid_2} words and semantic domain questions bidirectionally',
                total=len(self.verses_by_bid[bid_1])):
            if alignment_line in ('\n', '0-0\n') and len(wtxts_1) * len(wtxts_2) == 0:
                continue
            aligned_wtxt_pairs = alignment_line.split(' ')
            aligned_wtxt_pairs[-1].replace('\n', '')

            for aligned_wtxt_pair in aligned_wtxt_pairs:
                wtxt_1_idx, wtxt_2_idx = [int(num) for num in aligned_wtxt_pair.split('-')]
                wtxt_1 = wtxts_1[wtxt_1_idx]
                wtxt_2 = wtxts_2[wtxt_2_idx]

                # add alignment edge
                self._add_bidirectional_edge(wtxt_1, wtxt_2, lang_1, lang_2)

                # add qids
                new_qids_1 = self.words_by_text_by_lang[lang_1][wtxt_1].qids
                for new_qid_1 in new_qids_1:
                    self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2][lang_1][new_qid_1] += ', ' + wtxt_2

                new_qids_2 = self.words_by_text_by_lang[lang_2][wtxt_2].qids
                for new_qid_2 in new_qids_2:
                    self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1][lang_2][new_qid_2] += ', ' + wtxt_1

    def _map_target_words_to_qids(self):
        # map words in all target language bibles to semantic domains
        self.aligned_wtxts_by_qid_by_lang_by_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2:
                    continue
                with open(f'{self.data_path}/diag_[{bid_1}]_[{bid_2}].align', 'r') as alignment_file:
                    alignment = alignment_file.readlines()
                    self._map_two_bibles(alignment, bid_1, bid_2)

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
        for target_lang in self.target_langs:

            # merge alignments from all languages together
            merged_alignments = defaultdict(str)
            for lang in self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang]:
                for qid in self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][lang]:
                    merged_alignments[qid] += self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][lang][qid]

            tfidfs = self.vectorizer.fit_transform(list(merged_alignments.values()))
            assert (tfidfs.shape[0] == len(merged_alignments))
            for idx, tfidf in tqdm(enumerate(tfidfs), desc=f'collecting top {target_lang} tf-idf scores',
                                   total=tfidfs.shape[0]):  # caution: might fail in the debugger
                qid = list(merged_alignments.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=['TF-IDF'])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                self.top_tfidfs_by_qid_by_lang[target_lang][qid] = df.head(20)

    def train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()
        self._build_top_tfidfs()

        # build self.top_qids_by_wtxt_by_lang
        self.top_qids_by_wtxt_by_lang = defaultdict(lambda: defaultdict(list))
        for target_lang in self.target_langs:
            for qid, tfidfs_df in self.top_tfidfs_by_qid_by_lang[target_lang].items():
                for wtxt, tfidf in zip(list(tfidfs_df.index.values), list(tfidfs_df['TF-IDF'])):
                    self.top_qids_by_wtxt_by_lang[target_lang][wtxt].append((qid, tfidf))

        if save:
            self._save_state()

    def _filter_target_sds_with_threshold(self):
        # remove all target wtxts with a TF-IDF value below a threshold
        threshold = 0.15
        filtered_target_wtxts_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_langs:
            for qid, tfidfs_df in self.top_tfidfs_by_qid_by_lang[target_lang].items():
                filtered_target_wtxts_by_qid_by_lang[target_lang][qid] = list(
                    tfidfs_df[tfidfs_df['TF-IDF'] > threshold].index.values)
        return filtered_target_wtxts_by_qid_by_lang

    def _compute_f1_score(self, predicted_target_wtxts_by_qid, target_lang):
        """
        Compute precision, recall, and F1 score to evaluate DC. This requires a ground-truth semantic domain
        dictionary for the target language.
        """
        num_positive_wtxts = 0
        num_true_positive_wtxts = 0
        gt_target_wtxts_by_qid = self._group_words_by_qid(self.words_by_text_by_lang[target_lang])

        if len(gt_target_wtxts_by_qid) == 0:
            print(
                f'Cannot compute F1 score etc. for {target_lang} because no ground-truth target semantic domains have been loaded')
            return

        for qid, wtxts in tqdm(predicted_target_wtxts_by_qid.items(),
                               desc=f'counting true positive words in {target_lang} semantic domains',
                               total=len(predicted_target_wtxts_by_qid)):
            num_positive_wtxts += len(wtxts)
            for wtxt in wtxts:
                num_true_positive_wtxts += wtxt in gt_target_wtxts_by_qid.get(qid, [])

        # # How many non-unique wtxts are in the ground-truth target semantic domains?
        # num_total_sd_source_wtxts = 0
        # num_total_sd_wtxts_in_source_verses = 0
        # for _, wtxts_for_question in tqdm(self..items(),
        #                                   desc=f'collecting words in {self.source_language} semantic domains',
        #                                   total=len(self.)):
        #     num_total_sd_source_wtxts += len(wtxts_for_question)
        #     overlap = wtxts_for_question & self.source_verse_wtxts_set
        #     num_total_sd_wtxts_in_source_verses += len(wtxts_for_question & self.source_verse_wtxts_set)

        # How many non-unique wtxts are in the ground-truth target semantic domains?
        num_total_gt_target_wtxts = 0
        num_total_gt_sd_wtxts_in_target_verses = 0
        # num_total_single_gt_sd_wtxts_in_target_verses = 0
        for wtxts_for_question in tqdm(gt_target_wtxts_by_qid.values(),
                                       desc=f'collecting words in {target_lang} semantic domains',
                                       total=len(gt_target_wtxts_by_qid)):
            num_total_gt_target_wtxts += len(wtxts_for_question)
            overlap = [wtxt for wtxt in wtxts_for_question if
                       wtxt in self.words_by_text_by_lang[target_lang]
                       and self.words_by_text_by_lang[target_lang][wtxt].appears_in_bible]
            num_total_gt_sd_wtxts_in_target_verses += len(overlap)
            # single_wtxts = [wtxt for wtxt in overlap if ' ' not in wtxt]
            # num_total_single_gt_sd_wtxts_in_target_verses += len(single_wtxts)

        # How many of the found target wtxts actually appear in the ground-truth set?
        precision = num_true_positive_wtxts / num_positive_wtxts
        print(f'precision: {precision:.3f} ({num_true_positive_wtxts} '
              f'out of {num_positive_wtxts} found {target_lang} semantic domain words are correct)')

        # How many of the target sd wtxts in the ground-truth set were actually found?
        recall = num_true_positive_wtxts / num_total_gt_target_wtxts
        print(f'recall: {recall:.3f} ({num_true_positive_wtxts} '
              f'out of {num_total_gt_target_wtxts} {target_lang} actual semantic domain words found)')

        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'F1: {f1:.3f}\n')

        # # How many of the source wtxts appear in the source verses?
        # source_wtxt_coverage = num_total_sd_wtxts_in_source_verses / num_total_sd_source_wtxts
        # print(f'Source wtxt coverage: {source_wtxt_coverage:.3f} ({num_total_sd_wtxts_in_source_verses} '
        #       f'out of {len(num_total_sd_source_wtxts)} {self.source_language} actual non-unique semantic domain words '
        #       f'also appear in the source verses)')

        # How many of the gt target wtxts appear in the target verses?
        target_wtxt_coverage = num_total_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        print(
            f'Ground truth target word coverage: {target_wtxt_coverage:.3f} ({num_total_gt_sd_wtxts_in_target_verses} '
            f'out of {num_total_gt_target_wtxts} {target_lang} actual non-unique semantic domain words '
            f'also appear in the target verses)')

        # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # was actually found?
        recall_adjusted = num_true_positive_wtxts / num_total_gt_sd_wtxts_in_target_verses
        print(f'recall*: {recall_adjusted:.3f} ({num_true_positive_wtxts} '
              f'out of {num_total_gt_sd_wtxts_in_target_verses} {target_lang} actual semantic domain words '
              f'- that also appear in the target verses - found)')

        f1_adjusted = 2 * (precision * recall_adjusted) / (precision + recall_adjusted)
        print(f'F1*: {f1_adjusted:.3f}\n')

        # optional: consider wtxt groups vs. single wtxts in calculation
        # # How many of the single gt target wtxts appear in the target verses?
        # target_wtxt_coverage = num_total_single_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        # print(f'Ground truth single target wtxt coverage: {target_wtxt_coverage:.3f} ({num_total_single_gt_sd_wtxts_in_target_verses} '
        #       f'out of {num_total_gt_target_wtxts} {self.target_language} actual non-unique semantic domain words '
        #       f'also appear in the target verses)')
        #
        # # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # # was actually found?
        # recall_adjusted2 = num_true_positive_wtxts / num_total_single_gt_sd_wtxts_in_target_verses
        # print(f'recall**: {recall_adjusted2:.3f} ({num_true_positive_wtxts} '
        #       f'out of {num_total_single_gt_sd_wtxts_in_target_verses} {self.target_language} actual single semantic domain words '
        #       f'- that also appear in the target verses - found)')
        #
        # f1_adjusted2 = 2 * (precision * recall_adjusted2) / (precision + recall_adjusted2)
        # print(f'F1**: {f1_adjusted2:.3f}')

    def _load_test_data(self, source_lang, target_lang):
        # load source and corresponding target wtxts from Purdue Team (ground truth data for dictionary creation)
        df_test = pd.read_csv(f'{self.data_path}/multilingual_semdom_dictionary.csv')
        df_test = df_test[[f'{source_lang}-000.txt', f'{target_lang}-000.txt']]
        df_test.columns = ['source_wtxt', 'target_wtxts']
        df_test = df_test[df_test['target_wtxts'].notna()]

        df_test['source_wtxt'] = df_test['source_wtxt'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test['target_wtxts'] = df_test['target_wtxts'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test = df_test.explode('source_wtxt').reset_index(drop=True)
        # df_test = df_test.groupby(['target_wtxts']).agg(list)
        return df_test

    def _compute_mean_reciprocal_rank(self, target_lang):
        # compute MRR to evaluate DC
        # Filter target question that we are going to check because ground truth set is limited:
        # We only consider questions which have at least one source wtxt in the gt set with a target translation.
        # todo: Also filter out source wtxts (and questions, if empty) that do not appear in the source verses. (e.g., 'snake' does not appear in the KJV bible)
        if target_lang == 'urd':
            return None
        target_qids = defaultdict(list)
        source_lang = 'eng'  # todo: also do this for other source_langs
        df_test = self._load_test_data(source_lang, target_lang)
        for source_wtxt in tqdm(self.words_by_text_by_lang[source_lang].values(),
                                desc=f'filtering {target_lang} question ids',
                                total=len(self.words_by_text_by_lang[source_lang])):
            target_wtxts = list(df_test.query(f'source_wtxt=="{source_wtxt.text}"')['target_wtxts'])
            if len(target_wtxts) == 0:
                continue
            target_wtxts = target_wtxts[0]
            for qid in source_wtxt.qids:
                if qid in self.top_tfidfs_by_qid_by_lang[target_lang]:
                    target_qids[qid].extend(target_wtxts)
                # some semantic domains are missing in the target sds because no aligned wtxts were found

        # in all selected target top_tfidfs, look for first ranked target wtxt that also appears in df_test (gt data)
        mean_reciprocal_rank = 0
        for qid, target_wtxts in target_qids.items():
            wtxt_list = list(self.top_tfidfs_by_qid_by_lang[target_lang][qid].index)
            reciprocal_rank = 0
            for idx, wtxt in enumerate(wtxt_list):
                if wtxt in target_wtxts:
                    reciprocal_rank = 1 / (idx + 1)
                    break
            # print(qid)
            # print('\t', self.question_by_qid_by_lang[source_lang][qid])
            # print('\t found (positive) words:', wtxt_list)
            # print('\t reference (true) words:', target_wtxts)
            # print('\t reciprocal rank:       ', f'{reciprocal_rank:.2f}\n')
            mean_reciprocal_rank += reciprocal_rank
        mean_reciprocal_rank /= len(target_qids)
        print(
            f'{len(target_qids)} of {len(self.top_tfidfs_by_qid_by_lang[target_lang])} {target_lang} questions selected')
        print(f'MRR: {mean_reciprocal_rank:.3f}')

    def evaluate(self, load=False):
        if load:
            self._load_state()
        filtered_target_wtxts_by_qid_by_lang = self._filter_target_sds_with_threshold()
        for target_lang in self.target_langs:
            self._compute_f1_score(filtered_target_wtxts_by_qid_by_lang[target_lang], target_lang)
            self._compute_mean_reciprocal_rank(target_lang)


if __name__ == '__main__':
    dc = DictionaryCreator()
    dc.preprocess_data(save=True)
    dc.train_tfidf_based_model(load=True, save=True)
    dc.evaluate(load=True)
