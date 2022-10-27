import math
import os
import re
import subprocess
import time
from collections import defaultdict
from pickle import UnpicklingError

import dill
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError
from nltk.metrics.distance import edit_distance
from polyglot.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from word import Word


class DictionaryCreator(object):
    BIBLES_BY_BID = {
        'bid-eng-asvbt': 'eng-engasvbt.txt',
        'bid-eng-asv': 'eng-eng-asv.txt',
        'bid-eng-BBE': 'eng-engBBE.txt',
        'bid-eng-Brenton': 'eng-eng-Brenton.txt',
        'bid-eng-DBY': 'eng-engDBY.txt',
        'bid-eng-DRA': 'eng-engDRA.txt',
        'bid-eng-gnv': 'eng-enggnv.txt',
        'bid-eng-jps': 'eng-engjps.txt',
        'bid-eng-kjv2006': 'eng-eng-kjv2006.txt',
        'bid-eng-kjvcpb': 'eng-engkjvcpb.txt',
        'bid-eng-kjv': 'eng-eng-kjv.txt',
        'bid-eng-lee': 'eng-englee.txt',
        'bid-eng-lxx2012': 'eng-eng-lxx2012.txt',
        'bid-eng-lxxup': 'eng-englxxup.txt',
        'bid-eng-noy': 'eng-engnoy.txt',
        'bid-eng-oebcw': 'eng-engoebcw.txt',
        'bid-eng-oebus': 'eng-engoebus.txt',
        'bid-eng-oke': 'eng-engoke.txt',
        'bid-eng-rv': 'eng-eng-rv.txt',
        'bid-eng-tnt': 'eng-engtnt.txt',
        'bid-eng-uk-lxx2012': 'eng-eng-uk-lxx2012.txt',
        'bid-eng-webbe': 'eng-eng-webbe.txt',
        'bid-eng-web-c': 'eng-eng-web-c.txt',
        'bid-eng-webpb': 'eng-engwebpb.txt',
        'bid-eng-webp': 'eng-engwebp.txt',
        'bid-eng-webster': 'eng-engwebster.txt',
        'bid-eng-web': 'eng-eng-web.txt',
        'bid-eng-wmbb': 'eng-engwmbb.txt',
        'bid-eng-wmb': 'eng-engwmb.txt',
        'bid-eng-Wycliffe': 'eng-engWycliffe.txt',
        'bid-eng-ylt': 'eng-engylt.txt',
        'bid-eng-niv11': 'extra_english_bibles/en-NIV11.txt',
        'bid-eng-niv84': 'extra_english_bibles/en-NIV84.txt',
        'bid-eng-REB89': 'extra_english_bibles/en-REB89.txt',  # mentions "Euphrates" 65 times

        'bid-fra-fob': 'fra-fra_fob.txt',
        'bid-fra-lsg': 'fra-fraLSG.txt',

        'bid-spa': 'spa-spaRV1909.txt',
        'bid-ind': 'ind-ind.txt',
        'bid-tel': 'tel-telirv.txt',
        'bid-tha': 'tha-thaKJV.txt',
        'bid-hin': 'hin-hinirv.txt',
        'bid-nep': 'nep-nepulb.txt',
        'bid-urd': 'urd-urdgvu.txt',

        'bid-deu': 'no semdoms available/deu-deuelo.txt',
        'bid-rus': 'no semdoms available/rus-russyn.txt',
        'bid-vie': 'no semdoms available/vie-vie1934.txt',
        'bid-tpi': 'no semdoms available/tpi-tpipng.txt',  # mentions "Yufretis" 65 times
        'bid-swp': 'no semdoms available/swp-swp.txt',
    }

    def __init__(self, bids, score_threshold=0.5,
                 state_files_path='data/0_state',
                 aligned_bibles_path='data/1_aligned_bibles',
                 sd_path_prefix='../semdom extractor/output/semdom_qa_clean'):
        self.bids = bids
        self.bibles_by_bid = {bid: DictionaryCreator.BIBLES_BY_BID[bid] for bid in bids}
        self.source_bid = self.bids[0]
        self.source_lang = self._convert_bid_to_lang(self.source_bid)
        self.target_langs = sorted(set([self._convert_bid_to_lang(bid) for bid in self.bids]))
        self.all_langs = sorted(
            ['eng', 'fra', 'spa', 'ind', 'deu', 'rus', 'tha', 'tel', 'urd', 'hin', 'nep', 'vie', 'tpi', 'swp'])
        self.state_files_path = state_files_path
        self.aligned_bibles_path = aligned_bibles_path
        self.tokenizer = 'bpe'
        self.eng_lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.state_loaded = False
        self.score_threshold = score_threshold
        self.sd_path_prefix = sd_path_prefix
        self.start_timestamp = time.time_ns() // 1000  # current time in microseconds
        self.num_verses = 41899

        # Saved data (preprocessing)
        self.sds_by_lang = {}
        self.verses_by_bid = {}
        self.words_by_text_by_lang = defaultdict(dict)
        self.question_by_qid_by_lang = defaultdict(dict)
        self.wtxts_by_verse_by_bid = {}

        # Saved data (mapping)
        self.aligned_wtxts_by_qid_by_lang_by_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

        # Saved data (graph building)
        self.word_graph = None

        # Saved data (lemmas)
        self.base_lemma_by_wtxt_by_lang = defaultdict(dict)
        self.lemma_group_by_base_lemma_by_lang = defaultdict(lambda: defaultdict(set))

        # Saved data (predicting links)
        self.strength_by_lang_by_wtxt_by_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Saved data (training)
        self.top_scores_by_qid_by_lang = defaultdict(dict)

        # Saved data (evaluation)
        self.evaluation_results_by_lang = defaultdict(dict)

        # Stores which variables have changed since they have last been saved to a file
        self.changed_variables = set()

    @staticmethod
    def _convert_bid_to_lang(bid):
        return bid[4:7]

    @staticmethod
    def _group_words_by_qid(words):
        wtxts_by_qid = defaultdict(list)
        for word in words.values():
            for qid in word.qids:
                wtxts_by_qid[qid].append(word.text)
        return wtxts_by_qid

    @staticmethod
    def _transliterate_word(word):
        if word.iso_language == 'hin':
            return ' '.join(Text(word.display_text, word.iso_language).transliterate('en'))
        return word.display_text

    @staticmethod
    def _apply_prediction(func, ebunch):
        """Applies the given function to each edge in the specified iterable
        of edges.
        """
        return ((u, v, func(u, v)) for u, v in ebunch)

    def _weighted_resource_allocation_index(self, ebunch):
        r"""Compute the weighted resource allocation index of all node pairs in ebunch.

        References
        ----------
        .. resource allocation... todo (mentioned on page 3 in https://www.nature.com/articles/srep12261.pdf) [1] T. Zhou, L. Lu, Y.-C. Zhang.
           Predicting missing links via local information.
           Eur. Phys. J. B 71 (2009) 623.
           https://arxiv.org/pdf/0901.0553.pdf
        """

        def predict(word_1, word_2):
            return sum((self.word_graph.get_edge_data(word_1, common_neighbor)['weight'] +
                        self.word_graph.get_edge_data(common_neighbor, word_2)['weight'])
                       / (self._compute_sum_of_weights(common_neighbor, word_1.iso_language) +
                          self._compute_sum_of_weights(common_neighbor, word_2.iso_language))
                       for common_neighbor in nx.common_neighbors(self.word_graph, word_1, word_2)
                       if
                       word_1.iso_language != common_neighbor.iso_language != word_2.iso_language)  # ignore eng-eng-eng edges

        return DictionaryCreator._apply_prediction(predict, ebunch)

    def _save_state(self):
        if len(self.changed_variables) == 0:
            return
        print('Saving state...')

        # save newly changed class variables to a separate dill file to speed up saving
        for variable_name in tqdm(self.changed_variables,
                                  desc='Saving class variables',
                                  total=len(self.changed_variables)):
            variable = getattr(self, variable_name)

            def save_file(key=''):
                if key:
                    file_path = os.path.join(self.state_files_path,
                                             f'{self.start_timestamp}_{variable_name}_{key}.dill')
                else:
                    file_path = os.path.join(self.state_files_path, f'{self.start_timestamp}_{variable_name}.dill')

                with open(file_path, 'wb') as state_file:
                    if key:
                        dill.dump(variable[key], state_file)
                    else:
                        dill.dump(variable, state_file)

            if type(variable) is dict or type(variable) is defaultdict:
                for key, value in tqdm(variable.items(),
                                       desc=f'Saving {variable_name}',
                                       total=len(variable),
                                       leave=True,
                                       position=0):
                    save_file(key)
            else:
                save_file()

        self.changed_variables.clear()
        print('State saved.')

    def _find_most_recent_files(self):
        # file format: {start_timestamp}_{variable_name}_{key}.dill
        file_names = os.listdir(os.path.join(self.state_files_path))
        timestamps = [file_name.split('_')[0] for file_name in file_names]
        timestamps.sort()
        if len(timestamps):
            most_recent_timestamp = timestamps[-1]

            # This dc should be newer than any other dc, and we do not need to load the own state.
            assert (most_recent_timestamp < str(self.start_timestamp))

        return [file_name for file_name in file_names if file_name.startswith(most_recent_timestamp)]

    def _load_state(self):
        if self.state_loaded:
            return

        print('Loading state...')

        most_recent_files = self._find_most_recent_files()

        # load class variables from separate dill files
        for variable_name in ['sds_by_lang', 'verses_by_bid', 'words_by_text_by_lang', 'question_by_qid_by_lang',
                              'wtxts_by_verse_by_bid', 'aligned_wtxts_by_qid_by_lang_by_lang', 'word_graph',
                              'base_lemma_by_wtxt_by_lang', 'lemma_group_by_base_lemma_by_lang',
                              'strength_by_lang_by_wtxt_by_lang', 'top_scores_by_qid_by_lang',
                              'evaluation_results_by_lang']:
            variable = getattr(self, variable_name)

            def load_file(fallback_value, file_path):
                try:
                    with open(file_path, 'rb') as state_file:
                        return dill.load(state_file)
                except (EOFError, UnpicklingError):
                    print(f'{file_path} is broken. Skipping.')
                    return fallback_value

            # get all matching file names in directory
            file_paths = [os.path.join(self.state_files_path, file_name) for file_name in most_recent_files if
                          '_'.join(file_name.split('_')[1:]).startswith(variable_name)]

            if type(variable) is dict or type(variable) is defaultdict:
                for file_path in file_paths:
                    key = file_path.split('_')[-1].split('.')[0]
                    assert (key not in ('lang', 'bid'))
                    if key in self.target_langs or key in self.bids:
                        variable[key] = load_file(None, file_path)
            else:
                if len(file_paths):
                    assert (len(file_paths) == 1)
                    setattr(self, variable_name, load_file(variable, file_paths[0]))

        # self.top_scores_by_qid_by_lang = defaultdict(dict)  # activate this to switch between computing link scores and tf-idf scores
        # self.base_lemma_by_wtxt_by_lang = defaultdict(dict)
        # self.lemma_group_by_base_lemma_by_lang = defaultdict(lambda: defaultdict(set))

        self.state_loaded = True
        print('State loaded.')

    def _load_data(self):
        # load sds and bible verses for all languages
        languages = set([self._convert_bid_to_lang(bid) for bid in self.bids])
        for lang in tqdm(languages, desc='Loading semantic domains', total=len(languages)):
            # load sds
            if lang in self.sds_by_lang:
                print(f'Skipped: SDs for {lang} already loaded')
                continue

            sd_path = f'{self.sd_path_prefix}_{lang}.csv'
            if os.path.isfile(sd_path):
                self.sds_by_lang[lang] = pd.read_csv(sd_path)
            else:
                print(f'WARNING: Unable to load {sd_path}')
                # create empty dataframe
                self.sds_by_lang[lang] = pd.DataFrame(
                    {'cid': [], 'category': [], 'question_index': [], 'question': [], 'answer': []})
                if lang not in ('deu', 'rus', 'vie', 'tpi'):
                    raise FileNotFoundError(f'Unable to load {sd_path}')
            self.changed_variables.add('sds_by_lang')

        for bid in tqdm(self.bids, desc='Loading bibles', total=len(self.bids)):
            # load bible verses
            if bid in self.verses_by_bid:
                print(f'Skipped: Bible {bid} already loaded')
                continue

            with open(os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid]),
                      'r') as bible:
                self.verses_by_bid[bid] = bible.readlines()
                self.changed_variables.add('verses_by_bid')
            assert (len(self.verses_by_bid[bid]) == self.num_verses)

    def _build_sds(self):
        # convert sd dataframe to dictionary
        # optional: increase performance by querying wtxts from words_eng
        # optional: increase performance by using dataframe instead of dict
        for lang, sds in tqdm(self.sds_by_lang.items(), desc='Building semantic domains', total=len(self.sds_by_lang)):
            if lang in self.words_by_text_by_lang:
                print(f'Skipped: Words and SD questions for {lang} already loaded')
                continue

            for index, row in sds.iterrows():
                question = row.question.replace("'", '')
                question = question.replace('"', '')
                answer = row.answer.replace("'", '')
                answer = answer.replace('"', '')
                qid = f'{row.cid} {row.question_index}'
                wtxts = [wtxt.strip().lower() for wtxt in answer.split(',') if wtxt]
                if lang == 'eng':
                    wtxts = self._lemmatize_english_verse(wtxts)
                words = {wtxt: Word(wtxt.strip(), lang, {qid}) for wtxt in wtxts}

                # add new words to words_by_text_by_lang
                for word in words.values():
                    if word.text not in self.words_by_text_by_lang[lang]:
                        self.words_by_text_by_lang[lang][word.text] = word
                    else:
                        self.words_by_text_by_lang[lang][word.text].qids.add(qid)
                    self.changed_variables.add('words_by_text_by_lang')

                self.question_by_qid_by_lang[lang][qid] = question
                self.changed_variables.add('question_by_qid_by_lang')

    def _lemmatize_english_verse(self, verse):
        # https://stackoverflow.com/a/57686805/8816968
        lemmatized_wtxts = []
        pos_labels = pos_tag(verse)
        for wtxt, pos_label in pos_labels:
            pos_label = pos_label[0].lower()
            if pos_label == 'j':
                pos_label = 'a'  # reassignment

            if pos_label == 'r':  # adverbs
                try:
                    pertainyms = wordnet.synset(wtxt + '.r.1').lemmas()[0].pertainyms()
                    if len(pertainyms):
                        lemmatized_wtxts.append(pertainyms[0].name())
                    else:
                        lemmatized_wtxts.append(wtxt)
                except WordNetError:
                    lemmatized_wtxts.append(wtxt)
            elif pos_label in ['a', 's', 'v']:  # adjectives and verbs
                lemmatized_wtxts.append(self.eng_lemmatizer.lemmatize(wtxt, pos=pos_label))
            else:  # nouns and everything else
                lemmatized_wtxts.append(self.eng_lemmatizer.lemmatize(wtxt))
        return lemmatized_wtxts

    def _tokenize_verses(self):
        for bid in tqdm(self.bids, desc='Tokenizing verses', total=len(self.bids)):
            if bid in self.wtxts_by_verse_by_bid:
                print(f'Skipped: Bible {bid} already tokenized')
                continue

            assert (self.tokenizer == 'bpe')
            file = os.path.join(
                '../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid])
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer()  # todo: fix tokenizer (e.g., splits 'Prahlerei' into 'Pra' and 'hlerei') (might not be so important because this mainly happens for rare words) possible solution: use pre-defined word list for English, using utoken instead did not significantly improve results
            # todo: try out a WordPieceTrainer (https://towardsdatascience.com/designing-tokenizers-for-low-resource-languages-7faa4ab30ef4)
            tokenizer.train(files=[file], trainer=trainer)

            # tokenize all verses
            wtxts_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_bid[bid]]

            # lowercase all wtxts
            wtxts_by_verse = [[wtxt.lower() for wtxt in verse] for verse in wtxts_by_verse]

            # lemmatize all English words
            lang = self._convert_bid_to_lang(bid)
            if lang == 'eng':
                wtxts_by_verse = [self._lemmatize_english_verse(verse) for verse in wtxts_by_verse]

            self.wtxts_by_verse_by_bid[bid] = wtxts_by_verse.copy()
            self.changed_variables.add('wtxts_by_verse_by_bid')

            # mark words as appearing in the bible
            wtxts = [wtxt for wtxts in wtxts_by_verse for wtxt in wtxts]
            for wtxt in wtxts:
                if wtxt in self.words_by_text_by_lang[lang]:
                    self.words_by_text_by_lang[lang][wtxt].occurrences_in_bible += 1
                else:
                    self.words_by_text_by_lang[lang][wtxt] = Word(wtxt, lang, set(), 1)
                self.changed_variables.add('words_by_text_by_lang')

    def _combine_alignments(self):
        # combine verses from two different bibles into a single file for wtxt aligner (fast_align)
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2 and not (bid_1 == self.source_bid and bid_2 == self.source_bid):
                    # map every pair of different bibles plus the source bible to the source bible
                    continue

                aligned_bibles_file_path = f'{self.aligned_bibles_path}/diag_{bid_1}_{bid_2}_{self.tokenizer}.align'
                if os.path.isfile(aligned_bibles_file_path):
                    print(f'Skipped: Aligned bibles file {aligned_bibles_file_path} already exists')
                    continue

                combined_bibles_file_path = f'{self.aligned_bibles_path}/{bid_1}_{bid_2}_{self.tokenizer}.txt'
                with open(combined_bibles_file_path, 'w') as combined_bibles:
                    for idx, (bid_1_wtxts, bid_2_wtxts) in tqdm(enumerate(
                            zip(self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2])),
                            desc=f'Combining alignments for {bid_1} and {bid_2}',
                            total=len(self.wtxts_by_verse_by_bid[bid_1])):
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0 and len(bid_1_wtxts) + len(bid_2_wtxts) > 0:
                            # verse is missing in only one bible
                            print('Missing verse - verses might be misaligned!', idx, bid_1_wtxts, bid_2_wtxts)
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0:
                            # verse is missing in both bibles
                            bid_1_wtxts = ['#placeholder#']
                            bid_2_wtxts = ['#placeholder#']
                        combined_bibles.write(' '.join(bid_1_wtxts) + ' ||| ' + ' '.join(bid_2_wtxts) + '\n')

                result = subprocess.run(
                    ['sh', 'align_bibles.sh', bid_1, bid_2, self.tokenizer, self.aligned_bibles_path],
                    capture_output=True, text=True)
                # retrieve the final entropy and perplexity
                matches = re.search(r'FINAL(.|\n)*cross entropy: (\d+\.\d+)\n *perplexity: (\d+\.\d+)', result.stderr)
                cross_entropy = float(matches.group(2))
                perplexity = float(matches.group(3))
                print(f'cross entropy: {cross_entropy}, perplexity: {perplexity}')

    def preprocess_data(self, load=False, save=False):
        if load:
            self._load_state()
        self._load_data()
        self._build_sds()
        self._tokenize_verses()
        self._combine_alignments()
        if save:
            self._save_state()

    def _add_bidirectional_edge(self, word_1, word_2, count=1):
        word_1.add_aligned_word(word_2, count)
        word_2.add_aligned_word(word_1, count)
        self.changed_variables.add('words_by_text_by_lang')

    def _map_word_to_qid(self, source_wtxt, target_wtxt, source_lang, target_lang, link_score=None,
                         score_by_wtxt_by_qid_by_lang=None):
        for new_qid in self.words_by_text_by_lang[source_lang][source_wtxt].qids:
            if link_score is None:
                self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][source_lang][new_qid] += ', ' + target_wtxt
                self.changed_variables.add('aligned_wtxts_by_qid_by_lang_by_lang')
            else:
                score_by_wtxt = score_by_wtxt_by_qid_by_lang[target_lang][new_qid]
                if target_wtxt in score_by_wtxt:
                    score_by_wtxt[target_wtxt] = max(score_by_wtxt[target_wtxt],
                                                     link_score)  # todo: find mathematically elegant solution than using just the highest link score (something like 0.7 and 0.3 --> 0.9)
                else:
                    score_by_wtxt[target_wtxt] = link_score

    def _map_word_to_qid_bidirectionally(self, wtxt_1, wtxt_2, lang_1, lang_2, link_score=None,
                                         score_by_wtxt_by_qid_by_lang=None):
        self._map_word_to_qid(wtxt_1, wtxt_2, lang_1, lang_2, link_score, score_by_wtxt_by_qid_by_lang)
        if lang_1 == lang_2 and wtxt_1 == wtxt_2:
            # only map word once onto itself (loop edge)
            return
        self._map_word_to_qid(wtxt_2, wtxt_1, lang_2, lang_1, link_score, score_by_wtxt_by_qid_by_lang)

    def _map_two_bibles(self, alignment, bid_1, bid_2):
        # map words in two bibles to semantic domains
        # Caveat: This function ignores wtxts that could not be aligned.
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)

        if lang_1 in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2] \
                or lang_2 in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1]:
            print(f'Skipped: {bid_1} and {bid_2} already mapped')
            return

        for alignment_line, wtxts_1, wtxts_2 in tqdm(
                zip(alignment, self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2]),
                desc=f'Map {bid_1} and {bid_2} words and semantic domain questions bidirectionally',
                total=len(self.verses_by_bid[bid_1])):
            if alignment_line in ('\n', '0-0\n') and len(wtxts_1) * len(wtxts_2) == 0:
                continue
            aligned_wtxt_pairs = alignment_line.split(' ')
            aligned_wtxt_pairs[-1].replace('\n', '')

            for aligned_wtxt_pair in aligned_wtxt_pairs:
                wtxt_1_idx, wtxt_2_idx = [int(num) for num in aligned_wtxt_pair.split('-')]
                wtxt_1 = wtxts_1[wtxt_1_idx]
                wtxt_2 = wtxts_2[wtxt_2_idx]
                word_1 = self.words_by_text_by_lang[lang_1][wtxt_1]
                word_2 = self.words_by_text_by_lang[lang_2][wtxt_2]
                self._map_word_to_qid_bidirectionally(wtxt_1, wtxt_2, lang_1, lang_2)
                self._add_bidirectional_edge(word_1, word_2)

    def map_words_to_qids(self, load=False, save=False):
        # map words in all target language bibles to semantic domains
        if load:
            self._load_state()

        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2 and not (bid_1 == self.source_bid and bid_2 == self.source_bid):
                    # map every pair of different bibles plus the source bible to the source bible
                    continue
                with open(f'{self.aligned_bibles_path}/diag_{bid_1}_{bid_2}_{self.tokenizer}.align',
                          'r') as alignment_file:
                    alignment = alignment_file.readlines()
                    self._map_two_bibles(alignment, bid_1, bid_2)

        if save:
            self._save_state()

    def build_word_graph(self, load=False, save=False):
        if load:
            self._load_state()

        # flatmap words
        word_nodes = [word for lang in self.words_by_text_by_lang
                      if lang in self.target_langs  # ignore additional languages in graph
                      for word in self.words_by_text_by_lang[lang].values()]

        # get all edges for alignments between words in flat list
        weighted_edges = set()
        for word_1 in word_nodes:
            for word_2, count in word_1.get_aligned_words_and_counts(self.words_by_text_by_lang):
                if word_2 not in word_nodes:
                    continue
                weighted_edges.add((word_1, word_2, count))

        # create graph structures with NetworkX
        self.word_graph = nx.Graph()
        self.word_graph.add_nodes_from(word_nodes)
        self.word_graph.add_weighted_edges_from(weighted_edges)
        self.changed_variables.add('word_graph')

        if save:
            self._save_state()

    def plot_subgraph(self, lang, text, min_count=1):
        filtered_word_nodes = [word for word in self.word_graph.nodes if word.iso_language in self.target_langs]

        filtered_weighted_edges = []
        for edge in self.word_graph.edges(data='weight'):
            lang_1 = edge[0].iso_language
            lang_2 = edge[1].iso_language
            wtxt_1 = edge[0].text
            wtxt_2 = edge[1].text
            count = edge[2]
            if lang_1 not in self.target_langs or lang_2 not in self.target_langs \
                    or (lang_1 == lang_2 and wtxt_1 == wtxt_2) \
                    or (count < min_count and self._compute_link_score(edge[0], edge[1]) < self.score_threshold):
                continue
            filtered_weighted_edges.append(edge)

        filtered_word_graph = nx.Graph()
        filtered_word_graph.add_nodes_from(filtered_word_nodes)
        filtered_word_graph.add_weighted_edges_from(filtered_weighted_edges)

        # define filtered subgraph of neighbors of neighbors of node
        node = self.words_by_text_by_lang[lang][text]
        selected_nodes = {node}
        neighbors_1st_order = set()
        neighbors_2nd_order = set()
        neighbors_3rd_order = set()
        for neighbor_1st_order in filtered_word_graph.neighbors(node):
            neighbors_1st_order.add(neighbor_1st_order)
            is_predicted_link_1st_order = self._compute_link_score(node, neighbor_1st_order) >= self.score_threshold
            for neighbor_2nd_order in filtered_word_graph.neighbors(neighbor_1st_order):
                if not is_predicted_link_1st_order \
                        and self._compute_link_score(neighbor_1st_order, neighbor_2nd_order) < self.score_threshold:
                    continue
                neighbors_2nd_order.add(neighbor_2nd_order)
                for neighbor_3rd_order in filtered_word_graph.neighbors(neighbor_2nd_order):
                    if self._compute_link_score(neighbor_2nd_order, neighbor_3rd_order) < self.score_threshold:
                        continue
                    neighbors_3rd_order.add(neighbor_3rd_order)
        # avoid that graph gets too large or messy for plotting
        max_nodes = 50
        selected_nodes.update(neighbors_1st_order)
        if len(selected_nodes) + len(neighbors_2nd_order) <= max_nodes:
            selected_nodes.update(neighbors_2nd_order)
            if len(selected_nodes) + len(neighbors_3rd_order) <= max_nodes:
                selected_nodes.update(neighbors_3rd_order)
        displayed_subgraph = filtered_word_graph.subgraph(selected_nodes)
        assert (len(displayed_subgraph.nodes) <= len(
            displayed_subgraph.edges) + 1)  # necessary condition if graph is connected

        # set figure size heuristically
        width = max(6, int(len(selected_nodes) / 2.2))
        plt.figure(figsize=(width, width))

        # use a different node color for each language
        palette = sns.color_palette('pastel')  # ('hls', len(self.target_langs))
        palette += [palette[2]] * 10  # hack to add color for 'vie' that is different from 'eng'
        palette = {lang: color for lang, color in zip(self.all_langs, palette)}
        node_colors = [palette[word.iso_language] for word in displayed_subgraph.nodes()]

        # show all the colors in a legend
        plt.legend(handles=[mpatches.Patch(color=palette[lang], label=lang) for lang in self.target_langs])

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(displayed_subgraph)

        # draw nodes
        nx.draw_networkx_nodes(displayed_subgraph, pos=pos, node_color=node_colors)

        # draw only word texts as node labels
        nx.draw_networkx_labels(displayed_subgraph, pos=pos,
                                labels={word: self._transliterate_word(word) for word in displayed_subgraph.nodes()})

        # draw edges (thicker edges for more frequent alignments)
        for edge in displayed_subgraph.edges(data='weight'):
            link_score = self._compute_link_score(edge[0], edge[1])
            color = 'green' if link_score >= self.score_threshold else 'gray'
            nx.draw_networkx_edges(displayed_subgraph, pos=pos, edgelist=[edge],
                                   width=[math.log(edge[2]) + 1], alpha=0.5,
                                   edge_color=color)

        # draw edge labels
        edge_weights = nx.get_edge_attributes(displayed_subgraph, 'weight')
        if len(edge_weights):
            nx.draw_networkx_edge_labels(displayed_subgraph, pos, edge_labels=edge_weights)

        # plot the title
        title = f'Words that fast_align aligned with the {lang} word "{text}"'
        if min_count > 1:
            title += f' at least {min_count} times'
        plt.title(title)

        plt.show()

    def _find_lemma_link_candidates(self):
        # find all pairs of nodes in the word graph with a common neighbor
        # (relevant if using the weighted resource allocation index)
        link_candidates = set()
        words_in_target_langs = [word for lang in self.target_langs
                                 for word in self.words_by_text_by_lang[lang].values()]
        for word_1 in tqdm(words_in_target_langs, desc='Finding link candidates', total=len(words_in_target_langs)):
            for common_neighbor in self.word_graph.neighbors(word_1):
                for word_2 in self.word_graph.neighbors(common_neighbor):
                    if word_1 != word_2 \
                            and word_2.iso_language in self.target_langs \
                            and word_1.iso_language == word_2.iso_language \
                            and (word_2, word_1) not in link_candidates:
                        link_candidates.add((word_1, word_2))
        return list(link_candidates)

    def _find_translation_link_candidates(self):
        # find all neighboring nodes in the word graph
        link_candidates = set()
        words_in_target_langs = [word for lang in self.target_langs
                                 for word in self.words_by_text_by_lang[lang].values()]
        for word_1 in tqdm(words_in_target_langs, desc='Finding link candidates', total=len(words_in_target_langs)):
            for word_2 in self.word_graph.neighbors(word_1):
                if word_1 != word_2 \
                        and word_2.iso_language in self.target_langs \
                        and word_1.iso_language != word_2.iso_language \
                        and (word_2, word_1) not in link_candidates:
                    link_candidates.add((word_1, word_2))
        return list(link_candidates)

    def _compute_sum_of_weights(self, word_1, lang_2):
        # compute sum of weights from word_1 to target_lang words in neighbors of word_1
        # precompute strength sums to improve performance --> store them in cache self.strength_by_lang_by_wtxt_by_lang
        if lang_2 not in self.strength_by_lang_by_wtxt_by_lang[word_1.iso_language][word_1.text]:
            self.strength_by_lang_by_wtxt_by_lang[word_1.iso_language][word_1.text][lang_2] = sum(
                self.word_graph.get_edge_data(word_1, w)['weight']
                for w in self.word_graph.neighbors(word_1)
                if w.iso_language == lang_2)
            self.changed_variables.add('strength_by_lang_by_wtxt_by_lang')
        return self.strength_by_lang_by_wtxt_by_lang[word_1.iso_language][word_1.text][lang_2]

    def _compute_link_score(self, word_1, word_2):
        # normalized edge weight = divide edge weight by the average sum of edge weights to words of the other language

        sum_weights_1_to_2 = self._compute_sum_of_weights(word_1, word_2.iso_language)
        sum_weights_2_to_1 = self._compute_sum_of_weights(word_2, word_1.iso_language)

        edge_weight = self.word_graph.get_edge_data(word_1, word_2)['weight']
        avg_weights_sum = (sum_weights_1_to_2 + sum_weights_2_to_1) / 2
        return edge_weight / avg_weights_sum

    def _predict_lemma_links(self, lemma_link_candidates):
        # preds = nx.jaccard_coefficient(self.word_graph)
        # preds = nx.adamic_adar_index(self.word_graph)
        preds = self._weighted_resource_allocation_index(lemma_link_candidates)
        for word_1, word_2, link_score in tqdm(preds, desc='Predicting lemma links', total=len(lemma_link_candidates)):
            assert (word_1.iso_language == word_2.iso_language)
            lang = word_1.iso_language
            wtxt_1 = word_1.text
            wtxt_2 = word_2.text

            if link_score < 0.01:
                continue
            distance = edit_distance(wtxt_1, wtxt_2)
            if distance < max(len(wtxt_1), len(wtxt_2)) / 3:
                # find the base lemma, which is the most frequent lemma
                base_lemma_1 = self.base_lemma_by_wtxt_by_lang[lang].get(wtxt_1, wtxt_1)
                base_lemma_2 = self.base_lemma_by_wtxt_by_lang[lang].get(wtxt_2, wtxt_2)
                words_by_text = self.words_by_text_by_lang[lang]

                # start with word_1 as the assumed base lemma
                new_base_lemma = wtxt_1
                if word_1.occurrences_in_bible < word_2.occurrences_in_bible:
                    # word_2 is more frequent than word_1
                    new_base_lemma = wtxt_2
                if words_by_text[new_base_lemma].occurrences_in_bible \
                        < words_by_text[base_lemma_1].occurrences_in_bible:
                    # word_1's base lemma is (even) more frequent
                    new_base_lemma = base_lemma_1
                if words_by_text[new_base_lemma].occurrences_in_bible \
                        < words_by_text[base_lemma_2].occurrences_in_bible:
                    # word_2's base lemma is (even) more frequent
                    new_base_lemma = base_lemma_2

                # build a group of all lemmas that belong together
                new_lemma_group = {wtxt_1, wtxt_2} \
                    .union(self.lemma_group_by_base_lemma_by_lang[lang][base_lemma_1]) \
                    .union(self.lemma_group_by_base_lemma_by_lang[lang][base_lemma_2])

                # update base lemmas and lemma groups
                # All lemmas point to the new base lemma. Only the base lemma points to the new lemma group.
                for wtxt in new_lemma_group:
                    self.base_lemma_by_wtxt_by_lang[lang][wtxt] = new_base_lemma
                    if wtxt in self.lemma_group_by_base_lemma_by_lang[lang]:
                        del self.lemma_group_by_base_lemma_by_lang[lang][wtxt]
                self.lemma_group_by_base_lemma_by_lang[lang][new_base_lemma] = new_lemma_group

        for lang in self.lemma_group_by_base_lemma_by_lang:
            # sort self.lemma_group_by_base_lemma_by_lang by key
            self.lemma_group_by_base_lemma_by_lang[lang] = dict(
                sorted(self.lemma_group_by_base_lemma_by_lang[lang].items(), key=lambda x: x[0]))

    def _predict_lemmas(self, load=False, save=False):
        if load:
            self._load_state()

        if len(self.base_lemma_by_wtxt_by_lang):
            print('Skipped: Lemmas were already predicted')
            assert (len(self.base_lemma_by_wtxt_by_lang) == len(self.target_langs) and
                    len(self.lemma_group_by_base_lemma_by_lang) == len(self.target_langs))
            return

        lemma_link_candidates = self._find_lemma_link_candidates()
        self._predict_lemma_links(lemma_link_candidates)

        # validate lemmas
        for lang in self.lemma_group_by_base_lemma_by_lang:
            for base_lemma, lemma_group in self.lemma_group_by_base_lemma_by_lang[lang].items():
                # check that all lemma groups contain at least the base lemma and another lemma
                assert (base_lemma in lemma_group)
                assert (len(lemma_group) > 1)

                # check that all lemmas point to the base lemma and that all non-base lemmas store no lemma group
                for lemma in lemma_group:
                    assert (self.base_lemma_by_wtxt_by_lang[lang][lemma] == base_lemma)
                    assert (lemma == base_lemma or lemma not in self.lemma_group_by_base_lemma_by_lang[lang])

        for lang in self.target_langs:
            # if we found no lemmas for a language, at least create an empty dictionary to show that we tried finding them
            if lang not in self.base_lemma_by_wtxt_by_lang:
                self.base_lemma_by_wtxt_by_lang[lang] = dict()
            if lang not in self.lemma_group_by_base_lemma_by_lang:
                self.lemma_group_by_base_lemma_by_lang[lang] = defaultdict(set)

        self.changed_variables.add('base_lemma_by_wtxt_by_lang')
        self.changed_variables.add('lemma_group_by_base_lemma_by_lang')

        if save:
            self._save_state()

    def _contract_lemmas(self, load=False, save=False):
        # merge lemmas in same lemma groups together into a single node
        if load:
            self._load_state()

        assert (len(self.base_lemma_by_wtxt_by_lang) == len(self.target_langs) and
                len(self.lemma_group_by_base_lemma_by_lang) == len(self.target_langs))

        # check if we already contracted the lemmas for at least one target language (except English)
        for lang in self.target_langs:
            if lang == 'eng':
                continue
            sample_lemma_wtxt_group = next(iter(self.lemma_group_by_base_lemma_by_lang[lang].values()))
            if any(wtxt not in self.words_by_text_by_lang[lang] for wtxt in sample_lemma_wtxt_group):
                # there exists one lemma wtxt in sample_lemma_wtxt_group that is not in self.words_by_text_by_lang
                print('Skipped: Lemma groups were already contracted')
                return

        for lang in self.lemma_group_by_base_lemma_by_lang:
            if lang == 'eng':
                continue  # todo?: We lemmatize English words using wordnet instead. --> todo: do not collect lemmas for English.

            for base_lemma_wtxt, lemma_wtxt_group in tqdm(self.lemma_group_by_base_lemma_by_lang[lang].items(),
                                                          desc=f'Contracting lemmas for {lang}',
                                                          total=len(self.lemma_group_by_base_lemma_by_lang[lang])):
                assert (len(lemma_wtxt_group) > 1)

                # collect words that belong to the same lemma group
                # by finding the corresponding words for each lemma text
                base_lemma_word = self.words_by_text_by_lang[lang][base_lemma_wtxt]
                lemma_group_words = set()
                for lemma_wtxt in lemma_wtxt_group:
                    if lemma_wtxt != base_lemma_wtxt:
                        lemma_group_words.add(self.words_by_text_by_lang[lang][lemma_wtxt])

                # contract words in the graph (i.e., merge all grouped lemma nodes into a single lemma group node)
                base_lemma_word.merge_words(lemma_group_words, self.words_by_text_by_lang,
                                            self.strength_by_lang_by_wtxt_by_lang, self.changed_variables)
                self.words_by_text_by_lang[lang][base_lemma_wtxt] = base_lemma_word
        self.changed_variables.add('words_by_text_by_lang')

        if save:
            self._save_state()

    def predict_links(self, load=False, save=False):
        if load:
            self._load_state()

        link_candidates = self._find_translation_link_candidates()

        score_by_wtxt_by_qid_by_lang = defaultdict(lambda: defaultdict(dict))
        for word_1, word_2 in tqdm(link_candidates, desc='Predicting links', total=len(link_candidates)):
            link_score = self._compute_link_score(word_1, word_2)
            self._map_word_to_qid_bidirectionally(word_1.text, word_2.text, word_1.iso_language, word_2.iso_language,
                                                  link_score, score_by_wtxt_by_qid_by_lang)

        for target_lang in self.target_langs:
            if target_lang in self.top_scores_by_qid_by_lang:
                print(f'Skipped: top {target_lang} scores already collected')
                continue

            score_by_wtxt_by_qid = score_by_wtxt_by_qid_by_lang[target_lang]
            for qid, score_by_wtxt in tqdm(score_by_wtxt_by_qid.items(),
                                           desc=f'Collecting top {target_lang} scores',
                                           total=len(score_by_wtxt_by_qid)):
                score_by_wtxt = dict(sorted(score_by_wtxt.items(), key=lambda x: x[1], reverse=True))
                self.top_scores_by_qid_by_lang[target_lang][qid] = score_by_wtxt
                self.changed_variables.add('top_scores_by_qid_by_lang')

        if save:
            self._save_state()

    def train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()

        for target_lang in self.target_langs:
            if target_lang in self.top_scores_by_qid_by_lang:
                print(f'Skipped: top {target_lang} tfidfs already collected')
                continue

            aligned_wtxts_by_qid = self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][self.source_lang]
            tfidfs = self.vectorizer.fit_transform(list(aligned_wtxts_by_qid.values()))
            assert (tfidfs.shape[0] == len(aligned_wtxts_by_qid))
            for idx, tfidf in tqdm(enumerate(tfidfs),
                                   desc=f'Collecting top {target_lang} tf-idf scores',
                                   total=tfidfs.shape[0]):
                qid = list(aligned_wtxts_by_qid.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=['TF-IDF'])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                df = df.head(20)

                # convert df to dict
                scores_by_wtxt = {word: score for word, score in zip(df.index, df['TF-IDF'])}

                self.top_scores_by_qid_by_lang[target_lang][qid] = scores_by_wtxt
                self.changed_variables.add('top_scores_by_qid_by_lang')

        if save:
            self._save_state()

    def _filter_target_sds_with_threshold(self):
        # remove all target wtxts with a score (e.g., TF-IDF) below a threshold
        filtered_target_wtxts_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_langs:
            for qid, score_by_wtxt in self.top_scores_by_qid_by_lang[target_lang].items():
                filtered_target_wtxts_by_qid_by_lang[target_lang][qid] = [key for key, value in score_by_wtxt.items() if
                                                                          value >= self.score_threshold]
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
            print(f'Cannot compute F1 score etc. for {target_lang} '
                  f'because no ground-truth target semantic domains have been loaded')
            return None

        false_positives = []
        false_negatives = []
        false_negatives_in_verses = []
        for qid, wtxts in tqdm(predicted_target_wtxts_by_qid.items(),
                               desc=f'Counting true positive words in {target_lang} semantic domains',
                               total=len(predicted_target_wtxts_by_qid),
                               disable=True):
            num_positive_wtxts += len(wtxts)
            for wtxt in wtxts:
                num_true_positive_wtxts += wtxt in gt_target_wtxts_by_qid.get(qid, [])

            new_false_positives = list(set(wtxts) - set(gt_target_wtxts_by_qid.get(qid, [])))
            new_false_positives = [(wtxt, self.words_by_text_by_lang[target_lang][wtxt].occurrences_in_bible) for wtxt
                                   in new_false_positives]
            # sort by number of occurrences in bible
            new_false_positives = sorted(new_false_positives, key=lambda x: x[1], reverse=True)
            false_positives.append((qid, new_false_positives))

            new_false_negatives = list(set(gt_target_wtxts_by_qid.get(qid, [])) - set(wtxts))
            false_negatives.append((qid, new_false_negatives))
            new_false_negatives_in_verses = [(wtxt, self.words_by_text_by_lang[target_lang][wtxt].occurrences_in_bible)
                                             for wtxt in new_false_negatives
                                             if wtxt in self.words_by_text_by_lang[target_lang]
                                             and self.words_by_text_by_lang[target_lang][wtxt].occurrences_in_bible]
            # sort by number of occurrences in verses
            new_false_negatives_in_verses = sorted(new_false_negatives_in_verses, key=lambda x: x[1], reverse=True)
            false_negatives_in_verses.append((qid, new_false_negatives_in_verses))

        # sort false matches (by qid) by number of false matches
        false_positives.sort(key=lambda x: len(x[1]), reverse=True)
        false_negatives.sort(key=lambda x: len(x[1]), reverse=True)
        false_negatives_in_verses.sort(key=lambda x: len(x[1]), reverse=True)

        # number of all false matches to facilitate error analysis during debugging
        num_false_positives = sum([len(x[1]) for x in false_positives])
        num_false_negatives = sum([len(x[1]) for x in false_negatives])
        num_false_negatives_in_verses = sum([len(x[1]) for x in false_negatives_in_verses])

        # # How many non-unique wtxts are in the ground-truth target semantic domains?
        # num_total_sd_source_wtxts = 0
        # num_total_sd_wtxts_in_source_verses = 0
        # for _, wtxts_for_question in tqdm(self..items(),
        #                                   desc=f'Collecting words in {self.source_language} semantic domains',
        #                                   total=len(self.),
        #                                   disable=True):
        #     num_total_sd_source_wtxts += len(wtxts_for_question)
        #     overlap = wtxts_for_question & self.source_verse_wtxts_set
        #     num_total_sd_wtxts_in_source_verses += len(wtxts_for_question & self.source_verse_wtxts_set)

        # How many non-unique wtxts are in the ground-truth target semantic domains?
        num_total_gt_target_wtxts = 0
        num_total_gt_sd_wtxts_in_target_verses = 0
        # num_total_single_gt_sd_wtxts_in_target_verses = 0
        for wtxts_for_question in tqdm(gt_target_wtxts_by_qid.values(),
                                       desc=f'Collecting words in {target_lang} semantic domains',
                                       total=len(gt_target_wtxts_by_qid),
                                       disable=True):
            num_total_gt_target_wtxts += len(wtxts_for_question)
            overlap = [wtxt for wtxt in wtxts_for_question if
                       wtxt in self.words_by_text_by_lang[target_lang]
                       and self.words_by_text_by_lang[target_lang][wtxt].occurrences_in_bible]
            num_total_gt_sd_wtxts_in_target_verses += len(overlap)
            # single_wtxts = [wtxt for wtxt in overlap if ' ' not in wtxt]
            # num_total_single_gt_sd_wtxts_in_target_verses += len(single_wtxts)

        # How many of the found target wtxts actually appear in the ground-truth set?
        precision = num_true_positive_wtxts / num_positive_wtxts
        print(f'precision: {precision:.3f} ({num_true_positive_wtxts} '
              f'/ {num_positive_wtxts} found {target_lang} semantic domain words are correct)')

        # How many of the target sd wtxts in the ground-truth set were actually found?
        recall = num_true_positive_wtxts / num_total_gt_target_wtxts
        print(f'recall:    {recall:.3f} ({num_true_positive_wtxts} '
              f'/ {num_total_gt_target_wtxts} {target_lang} actual semantic domain words found)')

        f1 = 0.0 if precision * recall == 0 else 2 * (precision * recall) / (precision + recall)
        print(f'F1:        {f1:.3f}')

        # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # was actually found?
        recall_adjusted = 0.0 if num_true_positive_wtxts == 0 else num_true_positive_wtxts / num_total_gt_sd_wtxts_in_target_verses
        print(f'recall*:   {recall_adjusted:.3f} ({num_true_positive_wtxts} '
              f'/ {num_total_gt_sd_wtxts_in_target_verses} {target_lang} actual semantic domain words '
              f'- that also appear in the target verses - found)')

        f1_adjusted = 0.0 if precision * recall_adjusted == 0 else 2 * (precision * recall_adjusted) / (
                    precision + recall_adjusted)
        print(f'F1*:       {f1_adjusted:.3f}\n')

        # How many of the gt target wtxts appear in the target verses?
        target_wtxt_coverage = num_total_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        print(
            f'Ground truth target word coverage: {target_wtxt_coverage:.3f} ({num_total_gt_sd_wtxts_in_target_verses} '
            f'/ {num_total_gt_target_wtxts} {target_lang} actual non-unique semantic domain words '
            f'also appear in the target verses)')

        # # How many of the source wtxts appear in the source verses?
        # source_wtxt_coverage = num_total_sd_wtxts_in_source_verses / num_total_sd_source_wtxts
        # print(f'Source wtxt coverage: {source_wtxt_coverage:.3f} ({num_total_sd_wtxts_in_source_verses} '
        #       f'/ {len(num_total_sd_source_wtxts)} {self.source_language} actual non-unique semantic domain words '
        #       'also appear in the source verses)')

        # optional: consider wtxt groups vs. single wtxts in calculation
        # # How many of the single gt target wtxts appear in the target verses?
        # target_wtxt_coverage = num_total_single_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        # print(f'Ground truth single target wtxt coverage: {target_wtxt_coverage:.3f} ({num_total_single_gt_sd_wtxts_in_target_verses} '
        #       f'/ {num_total_gt_target_wtxts} {self.target_language} actual non-unique semantic domain words '
        #       'also appear in the target verses)')
        #
        # # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # # was actually found?
        # recall_adjusted2 = num_true_positive_wtxts / num_total_single_gt_sd_wtxts_in_target_verses
        # print(f'recall**: {recall_adjusted2:.3f} ({num_true_positive_wtxts} '
        #       f'/ {num_total_single_gt_sd_wtxts_in_target_verses} {self.target_language} actual single semantic domain words '
        #       '- that also appear in the target verses - found)')
        #
        # f1_adjusted2 = 2 * (precision * recall_adjusted2) / (precision + recall_adjusted2)
        # print(f'F1**: {f1_adjusted2:.3f}')
        return precision, recall, f1, recall_adjusted, f1_adjusted

    def _load_test_data(self, source_lang, target_lang):
        # load source and corresponding target wtxts from Purdue Team (ground truth data for dictionary creation)
        df_test = pd.read_csv(f'data/multilingual_semdom_dictionary.csv')
        df_test = df_test[[f'{source_lang}-000.txt', f'{target_lang}-000.txt']]
        df_test.columns = ['source_wtxt', 'target_wtxts']
        df_test = df_test[df_test['target_wtxts'].notna()]

        df_test['source_wtxt'] = df_test['source_wtxt'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test['target_wtxts'] = df_test['target_wtxts'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test = df_test.explode('source_wtxt').reset_index(drop=True)
        # df_test = df_test.groupby(['target_wtxts']).agg(list)
        return df_test

    def _compute_mean_reciprocal_rank(self, target_lang, print_reciprocal_ranks=False):
        # compute MRR to evaluate DC
        # Filter target question that we are going to check because ground truth set is limited:
        # We only consider questions which have at least one source wtxt in the gt set with a target translation.
        # MRR improvement todo: Also filter out source wtxts (and questions, if empty) that do not appear in the source verses. (e.g., 'snake' does not appear in the KJV bible)
        # MRR improvement todo: also do this for source langs different from eng
        if target_lang in ('urd', 'tpi'):
            print(f'Cannot compute MRR for {target_lang} because no ground truth data is available.')
            return None
        target_qids = defaultdict(list)
        df_test = self._load_test_data(self.source_lang, target_lang)

        for _, row in tqdm(df_test.iterrows(),
                           desc=f'Filtering {target_lang} question ids',
                           total=len(df_test),
                           disable=True):
            source_word = self.words_by_text_by_lang[self.source_lang].get(row['source_wtxt'], None)
            if source_word is None:
                continue
            target_wtxts = row['target_wtxts']
            for qid in source_word.qids:
                if qid in self.top_scores_by_qid_by_lang[target_lang]:
                    target_qids[qid].extend(target_wtxts)
                # Some semantic domains are missing in the target sds because no aligned wtxts were found.

        # in all selected target top_scores, look for first ranked target wtxt that also appears in df_test (gt data)
        mean_reciprocal_rank = 0
        for qid, target_wtxts in target_qids.items():
            wtxt_list = list(self.top_scores_by_qid_by_lang[target_lang][qid])
            reciprocal_rank = 0
            for idx, wtxt in enumerate(wtxt_list):
                if wtxt in target_wtxts:
                    reciprocal_rank = 1 / (idx + 1)
                    break
            if print_reciprocal_ranks:
                print(qid)
                print('\t', self.question_by_qid_by_lang[self.source_lang][qid])
                print('\t found (positive) words:', wtxt_list)
                print('\t reference (true) words:', target_wtxts)
                print('\t reciprocal rank:       ', f'{reciprocal_rank:.2f}\n')
            mean_reciprocal_rank += reciprocal_rank
        mean_reciprocal_rank /= len(target_qids)
        print(
            f'{len(target_qids)} / {len(self.top_scores_by_qid_by_lang[target_lang])} {target_lang} questions selected')
        print(f'MRR: {mean_reciprocal_rank:.3f}')
        return mean_reciprocal_rank

    def evaluate(self, save=False, load=False, print_reciprocal_ranks=False):
        if load:
            self._load_state()

        filtered_target_wtxts_by_qid_by_lang = self._filter_target_sds_with_threshold()
        print(f'\'=== Bibles: {self.bids}, Threshold: {self.score_threshold} ===')
        for target_lang in self.target_langs:
            print(f'\n\n--- Evaluation for {target_lang} ---')
            predicted_target_wtxts_by_qid = filtered_target_wtxts_by_qid_by_lang[target_lang]
            if len(predicted_target_wtxts_by_qid) == 0:
                print(f'Cannot compute F1 score etc. and MRR for {target_lang} '
                      f'because no target semantic domains have been predicted. '
                      f'Have you loaded semantic domains for the source language?')
                continue
            precision, recall, f1, recall_adjusted, f1_adjusted = self._compute_f1_score(predicted_target_wtxts_by_qid,
                                                                                         target_lang)
            mean_reciprocal_rank = self._compute_mean_reciprocal_rank(target_lang, print_reciprocal_ranks)
            self.evaluation_results_by_lang[target_lang] = {
                'precision': precision,
                'recall': recall,
                'F1': f1,
                'recall*': recall_adjusted,
                'F1*': f1_adjusted,
                'MRR': mean_reciprocal_rank
            }
            self.changed_variables.add('evaluation_results_by_lang')

        if save:
            self._save_state()

    def create_dictionary(self, load=False, save=False, plot_word_lang='eng', plot_word='drink', min_count=1,
                          prediction_method='link prediction'):
        if prediction_method not in ('link prediction', 'tfidf'):
            raise NotImplementedError(f'Prediction method {prediction_method} not implemented.')

        self.preprocess_data(load=load, save=save)
        self.map_words_to_qids(load=load, save=save)

        self.build_word_graph(load=load, save=save)  # build the graph with single words as nodes
        self._predict_lemmas(load=load, save=save)
        self._contract_lemmas(load=load, save=save)
        self.build_word_graph(load=load, save=save)  # build the word graph with lemma groups as nodes
        if prediction_method == 'link prediction':
            self.predict_links(load=load, save=save)
        else:  # tfidf
            self.train_tfidf_based_model(load=load, save=save)
        self.plot_subgraph(lang=plot_word_lang, text=plot_word, min_count=min_count)

        self.evaluate(load=load, save=save, print_reciprocal_ranks=False)


if __name__ == '__main__':  # pragma: no cover
    dc = DictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
    dc.create_dictionary()
