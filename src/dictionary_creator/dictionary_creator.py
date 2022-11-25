import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pickle import UnpicklingError

import dill
import networkx as nx
import pandas as pd
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError
from polyglot.text import Text
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from src.word import Word


class DictionaryCreator(ABC):
    STEPS = None  # Implemented by child classes
    BIBLES_BY_BID = {
        'bid-eng-DBY-1000': '../../../dictionary_creator/test/data/eng-engDBY-1000-verses.txt',
        'bid-fra-fob-1000': '../../../dictionary_creator/test/data/fra-fra_fob-1000-verses.txt',

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
        self.state_loaded = False
        self.score_threshold = score_threshold
        self.sd_path_prefix = sd_path_prefix
        self.start_timestamp = time.time_ns() // 1000  # current time in microseconds
        self.num_verses = 41899

        # Saved data (general)
        self.progress_log = ['started']

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
        .. resource allocation... todo
           (mentioned on page 3 in https://www.nature.com/articles/srep12261.pdf) [1] T. Zhou, L. Lu, Y.-C. Zhang.

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
                       if word_1.iso_language != common_neighbor.iso_language != word_2.iso_language)  # ignore
            # eng-eng-eng edges

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

            def save_file(variable_key=''):
                if variable_key:
                    file_path = os.path.join(self.state_files_path,
                                             f'{self.start_timestamp}_{variable_name}_{variable_key}.dill')
                else:
                    file_path = os.path.join(self.state_files_path, f'{self.start_timestamp}_{variable_name}.dill')

                with open(file_path, 'wb') as state_file:
                    if variable_key:
                        dill.dump(variable[variable_key], state_file)
                    else:
                        dill.dump(variable, state_file)

            if type(variable) is dict or type(variable) is defaultdict:
                for key in tqdm(variable.keys(),
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
        most_recent_timestamp = None
        if len(timestamps):
            most_recent_timestamp = timestamps[-1]

            # This dc should be newer than any other dc, and we do not need to load the own state.
            assert (most_recent_timestamp < str(self.start_timestamp))

        return [file_name for file_name in file_names if file_name.startswith(most_recent_timestamp)]

    def _load_state(self):
        if self.state_loaded:
            return

        print('WARNING: Loading might cause inconsistent behavior. '
              'To get predictable results, you should execute the entire program without loading.')
        print('Loading state...')

        most_recent_files = self._find_most_recent_files()

        # load class variables from separate dill files
        for variable_name in ['progress_log', 'sds_by_lang', 'verses_by_bid', 'words_by_text_by_lang',
                              'question_by_qid_by_lang',
                              'wtxts_by_verse_by_bid', 'aligned_wtxts_by_qid_by_lang_by_lang', 'word_graph',
                              'base_lemma_by_wtxt_by_lang', 'lemma_group_by_base_lemma_by_lang',
                              'strength_by_lang_by_wtxt_by_lang', 'top_scores_by_qid_by_lang',
                              'evaluation_results_by_lang']:
            variable = getattr(self, variable_name)

            def load_file(fallback_value, path):
                try:
                    with open(path, 'rb') as state_file:
                        file_content = dill.load(state_file)
                        print(f'{path} loaded.')
                        return file_content
                except (EOFError, UnpicklingError):
                    print(f'{path} is broken. Skipping.')
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

        # self.top_scores_by_qid_by_lang = defaultdict(dict)  # activate this to switch between
        #   computing link scores and tf-idf scores

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
            file = os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid])
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()  # todo: try to delete this
            trainer = BpeTrainer()
            # todo: fix tokenizer (e.g., splits 'Prahlerei' into 'Pra' and 'hlerei') (might
            #   not be so important because this mainly happens for rare words) possible solution: use pre-defined word
            #   list for English, using utoken instead did not significantly improve results
            # todo: try out a WordPieceTrainer
            #   (https://towardsdatascience.com/designing-tokenizers-for-low-resource-languages-7faa4ab30ef4)
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

                aligned_bibles_file_path = f'{self.aligned_bibles_path}/{bid_1}_{bid_2}_{self.tokenizer}_diag.align'
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
                    ['sh', 'align_bibles.sh', bid_1, bid_2, self.tokenizer,
                     self.aligned_bibles_path],
                    capture_output=True, text=True)
                # retrieve the final entropy and perplexity
                matches = re.search(r'FINAL(.|\n)*cross entropy: (\d+\.\d+)\n *perplexity: (\d+\.\d+)', result.stderr)
                cross_entropy = float(matches.group(2))
                perplexity = float(matches.group(3))
                print(f'cross entropy: {cross_entropy}, perplexity: {perplexity}')

    def _check_already_done(self, target_progress, load):
        loaded_variables_by_step = {
            'started': [],
            '_preprocess_data': [],  # todo #12
            '_map_words_to_qids': [],
            '_build_word_graph (raw)': [],
            '_predict_lemmas': [],
            '_contract_lemmas': [],
            '_build_word_graph (contracted)': [],
            '_predict_translation_links': [],
            '_train_tfidf_based_model': [],
            '_evaluate': [],
        }

        if load:
            self._load_state()

        # assert a consistent order of progress steps
        for expected_step, actual_step in zip(self.progress_log, self.STEPS):
            assert (actual_step == expected_step, f'Expected {expected_step}, got {actual_step}')

        # assert that all variables are available that have been set until the last step
        assert (len(self.progress_log))
        last_step = self.progress_log[-1]
        for loaded_variable in loaded_variables_by_step[last_step]:
            assert (loaded_variable is not None)
            if type(loaded_variable) == dict:
                assert (len(loaded_variable) > 0)

        return target_progress in self.progress_log

    def _set_progress(self, progress, save):
        self.progress_log.append(progress)
        self.changed_variables.add('progress_log')
        if save:
            self._save_state()

    def execute_and_track_state(self, func, step_name=None, save=False, load=False, *args, **kwargs):
        if step_name is None:
            step_name = func.__name__
        if self._check_already_done(step_name, load):
            print(f'Skipped: {step_name} already done')
            return
        func(*args, **kwargs)
        self._set_progress(step_name, save)

    def _preprocess_data(self):
        self._load_data()
        self._build_sds()
        self._tokenize_verses()
        self._combine_alignments()

    def _add_bidirectional_edge(self, word_1, word_2, count=1):
        word_1.add_aligned_word(word_2, count)
        word_2.add_aligned_word(word_1, count)
        self.changed_variables.add('words_by_text_by_lang')

    def _map_word_to_qid(self, source_wtxt, target_wtxt, source_lang, target_lang, link_score=None,
                         score_by_wtxt_by_qid_by_lang=None):
        # map a target word to a qid by looking at the qids of the aligned source word
        for new_qid in self.words_by_text_by_lang[source_lang][source_wtxt].qids:
            if link_score is None:
                # map word for tf-idf based model
                self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][source_lang][new_qid] += ', ' + target_wtxt
                self.changed_variables.add('aligned_wtxts_by_qid_by_lang_by_lang')
            else:
                # map word for link prediction based model
                score_by_wtxt = score_by_wtxt_by_qid_by_lang[target_lang][new_qid]
                if target_wtxt in score_by_wtxt:
                    score_by_wtxt[target_wtxt] = max(score_by_wtxt[target_wtxt], link_score)
                    # todo: find mathematically more elegant solution than using just the highest link score
                    #  (something like 0.7 and 0.3 --> 0.9)
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
        # Caveat: This function ignores wtxts that could not have been aligned.
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)

        if lang_1 in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2] \
                or lang_2 in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1]:
            print(f'Skipped: {bid_1} and {bid_2} already mapped')  # todo: remove these skipped messages
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

    def _map_words_to_qids(self):
        # map words in all target language bibles to semantic domains
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2 and not (bid_1 == self.source_bid and bid_2 == self.source_bid):
                    # map every pair of different bibles plus the source bible to the source bible
                    continue
                with open(f'{self.aligned_bibles_path}/{bid_1}_{bid_2}_{self.tokenizer}_diag.align',
                          'r') as alignment_file:
                    alignment = alignment_file.readlines()
                    self._map_two_bibles(alignment, bid_1, bid_2)

    def _build_word_graph(self):
        # flatmap words
        word_nodes = [word for lang in self.words_by_text_by_lang
                      if lang in self.target_langs  # ignore additional languages in graph
                      for word in self.words_by_text_by_lang[lang].values()]

        # get all edges for alignments between words in flat list
        weighted_edges = set()
        for word_1 in word_nodes:
            for word_2, count in word_1.get_aligned_words_and_counts(self.words_by_text_by_lang):
                if word_2.iso_language not in self.target_langs:
                    continue
                weighted_edges.add((word_1, word_2, count))

        # create graph structures with NetworkX
        self.word_graph = nx.Graph()
        self.word_graph.add_nodes_from(word_nodes)
        self.word_graph.add_weighted_edges_from(weighted_edges)
        self.changed_variables.add('word_graph')

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
            return None, None, None, None, None

        false_positives = []
        false_negatives = []
        false_negatives_in_verses = []  # only false negatives that appear in at least one verse
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

        # find words that cause most false positive matches
        false_positives_by_wtxt = defaultdict(set)
        for qid, elements in false_positives:
            for wtxt, _2 in elements:
                false_positives_by_wtxt[wtxt].add(qid)
        false_positives_by_wtxt = sorted(false_positives_by_wtxt.items(), key=lambda x: len(x[1]), reverse=True)

        # find words that cause most false negative matches
        false_negatives_in_verses_by_wtxt = defaultdict(set)
        for qid, elements in false_negatives_in_verses:
            for wtxt, _2 in elements:
                false_negatives_in_verses_by_wtxt[wtxt].add(qid)
        false_negatives_in_verses_by_wtxt = sorted(false_negatives_in_verses_by_wtxt.items(), key=lambda x: len(x[1]),
                                                   reverse=True)

        # # number of all false matches to facilitate error analysis during debugging
        # num_false_positives = sum([len(x[1]) for x in false_positives])
        # num_false_negatives = sum([len(x[1]) for x in false_negatives])
        # num_false_negatives_in_verses = sum([len(x[1]) for x in false_negatives_in_verses])

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
        recall_adjusted = 0.0 if num_true_positive_wtxts == 0 \
            else num_true_positive_wtxts / num_total_gt_sd_wtxts_in_target_verses
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
        # print(f'Ground truth single target wtxt coverage: {target_wtxt_coverage:.3f} '
        #       f'({num_total_single_gt_sd_wtxts_in_target_verses} '
        #       f'/ {num_total_gt_target_wtxts} {self.target_language} actual non-unique semantic domain words '
        #       'also appear in the target verses)')
        #
        # # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # # was actually found?
        # recall_adjusted2 = num_true_positive_wtxts / num_total_single_gt_sd_wtxts_in_target_verses
        # print(f'recall**: {recall_adjusted2:.3f} ({num_true_positive_wtxts} '
        #       f'/ {num_total_single_gt_sd_wtxts_in_target_verses} {self.target_language} '
        #       f'actual single semantic domain words '
        #       '- that also appear in the target verses - found)')
        #
        # f1_adjusted2 = 2 * (precision * recall_adjusted2) / (precision + recall_adjusted2)
        # print(f'F1**: {f1_adjusted2:.3f}')
        return precision, recall, f1, recall_adjusted, f1_adjusted

    def _load_test_data(self, target_lang):
        # load source and corresponding target wtxts from Purdue Team (ground truth data for dictionary creation)
        df_test = pd.read_csv('data/multilingual_semdom_dictionary.csv')
        df_test = df_test[[f'{self.source_lang}-000.txt', f'{target_lang}-000.txt']]
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
        # MRR improvement todo: Also filter out source wtxts (and questions, if empty) that do not appear
        #   in the source verses. (e.g., 'snake' does not appear in the KJV bible)
        # MRR improvement todo: also do this for source langs different from eng
        if target_lang in ('urd', 'tpi'):
            print(f'Cannot compute MRR for {target_lang} because no ground truth data is available.')
            return None
        target_qids = defaultdict(list)
        df_test = self._load_test_data(target_lang)

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
        for qid, target_wtxts in sorted(target_qids.items()):  # sorting avoids numerical unreproducibility
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

    def _evaluate(self, print_reciprocal_ranks=False):
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

    @abstractmethod
    def create_dictionary(self, load=False, save=False, plot_word_lang='eng', plot_wtxt='drink', min_count=1):
        pass


if __name__ == '__main__':  # pragma: no cover
    # dc = DictionaryCreator(['bid-eng-DBY-1000', 'bid-fra-fob-1000'], score_threshold=0.2)
    dc = DictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
    dc.create_dictionary(load=True, save=True, plot_wtxt='river')
