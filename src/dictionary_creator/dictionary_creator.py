import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pickle import UnpicklingError

import dill
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

    LOADED_VARIABLES_BY_STEP = {
        '_preprocess_data': ['progress_log', 'sds_by_lang', 'verses_by_bid', 'words_by_text_by_lang',
                             'question_by_qid_by_lang', 'wtxts_by_verse_by_bid'],
        '_map_words_to_qids': ['aligned_wtxts_by_qid_by_lang_by_lang'],
        '_evaluate': ['evaluation_results_by_lang'],
    }

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
        self.state_files_base_path = state_files_path
        self.aligned_bibles_path = aligned_bibles_path
        self.tokenizer = 'bpe'
        self.eng_lemmatizer = WordNetLemmatizer()
        self.state_loaded = False
        self.score_threshold = score_threshold
        self.sd_path_prefix = sd_path_prefix
        self.start_timestamp = str(time.time_ns() // 1000)  # current time in microseconds
        self.num_verses = 41899

        # Saved data (general)
        self.progress_log = []  # all completed steps

        # Saved data (preprocessing)
        self.sds_by_lang = {}
        self.verses_by_bid = {}
        self.words_by_text_by_lang = defaultdict(dict)
        self.question_by_qid_by_lang = defaultdict(dict)
        self.wtxts_by_verse_by_bid = {}

        # Saved data (mapping)
        self.aligned_wtxts_by_qid_by_lang_by_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

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

    def _save_state(self):
        if len(self.changed_variables) == 0:
            return
        print('\nSaving state...')

        # create directory if it doesn't exist
        state_files_directory = os.path.join(self.state_files_base_path, self.start_timestamp)
        if not os.path.exists(state_files_directory):
            os.makedirs(state_files_directory)

        # save newly changed class variables to a separate dill file to speed up saving
        for variable_name in tqdm(self.changed_variables,
                                  desc='- Saving class variables',
                                  total=len(self.changed_variables)):
            variable = getattr(self, variable_name)

            def save_file(variable_key=''):
                if variable_key:
                    file_path = os.path.join(state_files_directory, f'{variable_name}_{variable_key}.dill')
                else:
                    file_path = os.path.join(state_files_directory, f'{variable_name}.dill')

                with open(file_path, 'wb') as state_file:
                    if variable_key:
                        dill.dump(variable[variable_key], state_file)
                    else:
                        dill.dump(variable, state_file)

            if type(variable) is dict or type(variable) is defaultdict:
                for key in tqdm(variable.keys(),
                                desc=f'- Saving {variable_name}',
                                total=len(variable),
                                leave=True,
                                position=0):
                    save_file(key)
            else:
                save_file()

        self.changed_variables.clear()
        print('State saved.\n')
        sys.stdout.flush()

    def _find_state_directory(self):
        # Directory names are timestamps. Find the most recent directory.
        timestamps = os.listdir(self.state_files_base_path)
        timestamps.sort()
        most_recent_timestamp = None
        if len(timestamps):
            most_recent_timestamp = timestamps[-1]

            # This dc should be newer than any other dc, and we do not need to load the own state.
            assert most_recent_timestamp < str(self.start_timestamp)
            self.start_timestamp = most_recent_timestamp
        return most_recent_timestamp

    def _load_state(self):
        # file path format: {state_files_base_path}/{start_timestamp}/{variable_name}_{key}.dill
        if self.state_loaded:
            return

        state_files_directory = self._find_state_directory()
        if state_files_directory is None:
            print('Skipped loading because no state files could be found.')
            self.state_loaded = True
            return

        print('Loading state...')
        state_files_directory = os.path.join(self.state_files_base_path, state_files_directory)
        file_names = os.listdir(state_files_directory)

        # create set of all variables that have been saved
        variable_names = set()
        for variables in self.LOADED_VARIABLES_BY_STEP.values():
            variable_names.update(variables)

        # load class variables from separate dill files
        for variable_name in variable_names:
            variable = getattr(self, variable_name)

            def load_file(fallback_value, path):
                try:
                    with open(path, 'rb') as state_file:
                        file_content = dill.load(state_file)
                        print(f'- {path} loaded.')
                        return file_content
                except (EOFError, UnpicklingError):
                    print(f'- {path} is broken. Skipping.')
                    return fallback_value

            # get all matching file names in directory
            file_names_for_variable = [os.path.join(state_files_directory, file_name) for file_name in file_names if
                                       file_name.startswith(variable_name)]

            if type(variable) is dict or type(variable) is defaultdict:
                for file_name in file_names_for_variable:
                    key = file_name.split('_')[-1].split('.')[0]
                    assert key not in ('lang', 'bid')
                    if key in self.target_langs or key in self.bids:
                        variable[key] = load_file(None, file_name)
            else:
                if len(file_names_for_variable):
                    assert len(file_names_for_variable) == 1
                    setattr(self, variable_name, load_file(variable, file_names_for_variable[0]))

        # self.top_scores_by_qid_by_lang = defaultdict(dict)  # activate this to switch between
        #   computing link scores and tf-idf scores

        self.state_loaded = True
        print('State loaded.')
        print('Progress:', self.progress_log, '\n')

    def _load_data(self):
        # load sds and bible verses for all languages
        languages = set([self._convert_bid_to_lang(bid) for bid in self.bids])
        for lang in tqdm(languages, desc='Loading semantic domains', total=len(languages)):
            # load sds
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
            with open(os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid]),
                      'r') as bible:
                self.verses_by_bid[bid] = bible.readlines()
                self.changed_variables.add('verses_by_bid')
            assert len(self.verses_by_bid[bid]) == self.num_verses

    def _build_sds(self):
        # convert sd dataframe to dictionary
        # optional: increase performance by querying wtxts from words_eng
        # optional: increase performance by using dataframe instead of dict
        for lang, sds in tqdm(self.sds_by_lang.items(), desc='Building semantic domains', total=len(self.sds_by_lang)):
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
            assert self.tokenizer == 'bpe'
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
        if load:
            self._load_state()

        # assert a consistent order of progress steps
        for expected_step, actual_step in zip(self.progress_log, self.STEPS):
            assert actual_step == expected_step

        # assert that all required variables are available
        for step in self.STEPS:
            if step == target_progress:
                break
            for loaded_variable_name in self.LOADED_VARIABLES_BY_STEP[step]:
                loaded_variable = getattr(self, loaded_variable_name)
                assert loaded_variable is not None
                if type(loaded_variable) in (dict, list, set, defaultdict):
                    assert len(loaded_variable) > 0, f'Loaded variable {loaded_variable_name} is empty'
                if loaded_variable_name.endswith('by_lang'):
                    assert len(loaded_variable) == len(self.target_langs)

        return target_progress in self.progress_log

    def _set_progress(self, progress, save):
        self.progress_log.append(progress)
        self.changed_variables.add('progress_log')
        if save:
            self._save_state()

    def _execute_and_track_state(self, func, step_name=None, save=False, load=False, *args, **kwargs):
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

    def _map_word_to_qids(self, source_wtxt, target_wtxt, source_lang, target_lang):
        """
        Assign a target word to all of a source word's qids.
        """
        for new_qid in self.words_by_text_by_lang[source_lang][source_wtxt].qids:
            self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][source_lang][new_qid] += ', ' + target_wtxt
            self.changed_variables.add('aligned_wtxts_by_qid_by_lang_by_lang')

    def _map_word_to_qid_bidirectionally(self, wtxt_1, wtxt_2, lang_1, lang_2):
        """
        Assign two words to all of each other's qids.
        """
        self._map_word_to_qids(wtxt_1, wtxt_2, lang_1, lang_2)
        if lang_1 == lang_2 and wtxt_1 == wtxt_2:
            return
        self._map_word_to_qids(wtxt_2, wtxt_1, lang_2, lang_1)

    def _map_two_bibles_bidirectionally(self, alignment, bid_1, bid_2):
        # map words in two bibles to semantic domains
        # Caveat: This function ignores wtxts that could not have been aligned.
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)

        # at least create an empty dictionary to show that we tried aligning words
        assert (lang_2 not in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1])
        assert (lang_1 not in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2])
        self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1][lang_2] = defaultdict(str)
        self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2][lang_1] = defaultdict(str)

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
                    self._map_two_bibles_bidirectionally(alignment, bid_1, bid_2)

    def _filter_target_sds_with_threshold(self):
        # remove all target wtxts with a score (e.g., TF-IDF) below a threshold
        filtered_target_wtxts_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_langs:
            for qid, score_by_wtxt in self.top_scores_by_qid_by_lang[target_lang].items():
                filtered_target_wtxts_by_qid_by_lang[target_lang][qid] = \
                    [(wtxt, (score, annotation)) for (wtxt, (score, annotation)) in score_by_wtxt.items() if
                     score >= self.score_threshold]
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
        for qid, annotated_wtxts in tqdm(predicted_target_wtxts_by_qid.items(),
                                         desc=f'Counting true positive words in {target_lang} semantic domains',
                                         total=len(predicted_target_wtxts_by_qid),
                                         disable=True):
            num_positive_wtxts += len(annotated_wtxts)
            for wtxt, annotation in annotated_wtxts:
                num_true_positive_wtxts += wtxt in gt_target_wtxts_by_qid.get(qid, [])

            annotation_by_wtxt = {wtxt: annotation for wtxt, annotation in annotated_wtxts}
            assert len(annotation_by_wtxt) == len(annotated_wtxts)

            new_false_positives = list(set(annotation_by_wtxt.keys()) - set(gt_target_wtxts_by_qid.get(qid, [])))
            new_false_positives = [
                (wtxt, self.words_by_text_by_lang[target_lang][wtxt].occurrences_in_bible, annotation_by_wtxt[wtxt])
                for wtxt in new_false_positives]
            # sort by number of occurrences in bible
            new_false_positives = sorted(new_false_positives, key=lambda x: x[1], reverse=True)
            false_positives.append((qid, new_false_positives))

            new_false_negatives = list(set(gt_target_wtxts_by_qid.get(qid, [])) - set(annotation_by_wtxt.keys()))
            false_negatives.append((qid, new_false_negatives))
            new_false_negatives_in_verses = [
                (wtxt, self.words_by_text_by_lang[target_lang][wtxt].occurrences_in_bible)
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
            for wtxt, _, annotation in elements:
                false_positives_by_wtxt[wtxt].add((qid, annotation))
        false_positives_by_wtxt = sorted(false_positives_by_wtxt.items(), key=lambda x: len(x[1]), reverse=True)

        # find words that cause most false negative matches
        false_negatives_in_verses_by_wtxt = defaultdict(set)
        for qid, elements in false_negatives_in_verses:
            for wtxt, _ in elements:
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
            wtxt_list = list(self.top_scores_by_qid_by_lang[target_lang][qid].keys())
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
        print(f'\n\n\'=== Bibles: {self.bids}, Threshold: {self.score_threshold} ===')
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
                'MRR': mean_reciprocal_rank,
            }
            self.changed_variables.add('evaluation_results_by_lang')

    @abstractmethod
    def create_dictionary(self, load=False, save=False, *args, **kwargs):  # pragma: no cover
        pass
