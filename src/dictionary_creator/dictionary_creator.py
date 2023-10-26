import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from datetime import datetime
from math import ceil
from pickle import UnpicklingError

import dill
import pandas as pd
import stanza
from fontTools import unicodedata
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
# from polyglot.text import Text
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import BertPreTokenizer
from tqdm import tqdm
from unidecode import unidecode

import src.utils
from src.dictionary_creator.find_stop_words import find_stop_words
from src.word import Word


class DictionaryCreator(ABC):
    STEPS = None  # Implemented by child classes

    STANZA_LANG_MAP = {
        # 'arb': 'ar',
        # 'pes': 'fa',
        # 'tel': 'te',
        # 'urd': 'ur',
        'eng': 'en',
        'deu': 'de',
        'fra': 'fr',
        'spa': 'es',
        'por': 'pt',
        'hin': 'hi',
        'ind': 'id',
        'cmn': 'zh-hans',
    }

    LOADED_VARIABLES_BY_STEP = {
        '_preprocess_data': ['progress_log', 'sds_by_lang', 'verses_by_bid', 'words_by_text_by_lang',
                             'question_by_qid_by_lang', 'wtxts_by_verse_by_bid'],
        '_map_words_to_qids': ['aligned_wtxts_by_qid_by_lang_by_lang'],
        '_remove_stop_words': [],
        '_evaluate': ['evaluation_results_by_lang'],
    }

    BIBLES_BY_BID = {
        # ENGLISH BIBLES
        'bid-eng-web': 'eBible/eng-eng-web.txt',  # with deuterocanon
        'bid-eng-rv': 'eBible/eng-eng-rv.txt',  # with deuterocanon, "_" removed
        'bid-eng-asvbt': 'eBible/eng-engasvbt.txt',  # with deuterocanon
        'bid-eng-asv': 'scripture_public_domain/eng-eng-asv.txt',
        'bid-eng-BBE': 'scripture_public_domain/eng-engBBE.txt',
        'bid-eng-Brenton': 'scripture_public_domain/eng-eng-Brenton.txt',
        'bid-eng-DBY': 'scripture_public_domain/eng-engDBY.txt',
        'bid-eng-DRA': 'scripture_public_domain/eng-engDRA.txt',
        'bid-eng-gnv': 'scripture_public_domain/eng-enggnv.txt',
        'bid-eng-jps': 'scripture_public_domain/eng-engjps.txt',
        'bid-eng-kjv2006': 'scripture_public_domain/eng-eng-kjv2006.txt',
        'bid-eng-kjvcpb': 'scripture_public_domain/eng-engkjvcpb.txt',
        'bid-eng-kjv': 'scripture_public_domain/eng-eng-kjv.txt',
        'bid-eng-lee': 'scripture_public_domain/eng-englee.txt',
        'bid-eng-lxx2012': 'scripture_public_domain/eng-eng-lxx2012.txt',
        'bid-eng-lxxup': 'scripture_public_domain/eng-englxxup.txt',
        'bid-eng-noy': 'scripture_public_domain/eng-engnoy.txt',
        'bid-eng-oebcw': 'scripture_public_domain/eng-engoebcw.txt',
        'bid-eng-oebus': 'scripture_public_domain/eng-engoebus.txt',
        'bid-eng-oke': 'scripture_public_domain/eng-engoke.txt',
        'bid-eng-tnt': 'scripture_public_domain/eng-engtnt.txt',
        'bid-eng-uk-lxx2012': 'scripture_public_domain/eng-eng-uk-lxx2012.txt',
        'bid-eng-webbe': 'scripture_public_domain/eng-eng-webbe.txt',
        'bid-eng-web-c': 'scripture_public_domain/eng-eng-web-c.txt',
        'bid-eng-webpb': 'scripture_public_domain/eng-engwebpb.txt',
        'bid-eng-webp': 'scripture_public_domain/eng-engwebp.txt',
        'bid-eng-webster': 'scripture_public_domain/eng-engwebster.txt',
        'bid-eng-wmbb': 'scripture_public_domain/eng-engwmbb.txt',
        'bid-eng-wmb': 'scripture_public_domain/eng-engwmb.txt',
        'bid-eng-Wycliffe': 'scripture_public_domain/eng-engWycliffe.txt',
        'bid-eng-ylt': 'scripture_public_domain/eng-engylt.txt',
        'bid-eng-niv11': 'scripture_public_domain/extra_english_bibles/en-NIV11.txt',
        'bid-eng-niv84': 'scripture_public_domain/extra_english_bibles/en-NIV84.txt',
        'bid-eng-REB89': 'scripture_public_domain/extra_english_bibles/en-REB89.txt',

        # OTHER BIBLE WITH SEMANTIC DOMAINS
        'bid-fra-fob': 'eBible/fra-fra_fob.txt',
        'bid-fra-lsg': 'eBible/fra-fraLSG.txt',
        'bid-fra-sbl': 'eBible/fra-frasbl.txt',  # with deuterocanon, NBSP removed
        'bid-spa-1909': 'eBible/spa-spaRV1909.txt',
        'bid-spa-blm': 'eBible/spa-spablm.txt',  # with deuterocanon
        'bid-por-bsl': 'eBible/por-porbrbsl.txt',
        'bid-por-2018': 'eBible/por-porbr2018.txt',
        'bid-swh-1850': 'eBible/swh-swh1850.txt',
        'bid-swh-onen': 'eBible/swh-swhonen.txt',
        'bid-swh-ulb': 'eBible/swh-swhulb.txt',
        'bid-ind': 'eBible/ind-ind.txt',
        'bid-tha': 'eBible/tha-thaKJV.txt',
        'bid-hin': 'eBible/hin-hin2017.txt',
        'bid-npi': 'eBible/npi-npiulb.txt',
        'bid-mya': 'eBible/mya-myajvb.txt',
        'bid-cmn-s': 'eBible/cmn-cmn-cu89s.txt',
        'bid-cmn-t': 'eBible/cmn-cmn-cu89t.txt',
        'bid-urd': 'eBible/urd-urdgvu.txt',
        'bid-ben': 'eBible/ben-ben2017.txt',
        'bid-mkn': 'eBible/mkn-mkn.txt',
        'bid-mal': 'eBible/mal-mal.txt',
        'bid-arb-vd': 'eBible/arb-arb-vd.txt',
        'bid-arb-nav': 'eBible/arb-arbnav.txt',
        'bid-tel': 'eBible/tel-tel2017.txt',

        # BIBLES WITHOUT SEMANTIC DOMAINS
        'bid-deu-elo': 'eBible/no semdoms available/deu-deuelo.txt',
        'bid-deu-1951': 'eBible/no semdoms available/deu-deu1951.txt',
        'bid-tpi': 'eBible/no semdoms available/tpi-tpi.txt',
        'bid-yor': 'eBible/no semdoms available/yor-yor.txt',
        'bid-meu': 'eBible/no semdoms available/meu-meu.txt',
        'bid-hmo': 'eBible/no semdoms available/hmo-hmo.txt',
        'bid-pes': 'eBible/pes-pesOPV.txt',
        'bid-azb': 'eBible/no semdoms available/azb-azb.txt',
        'bid-gej': 'scripture_copyrighted/no semdoms available/gej-GEN.txt',
        'bid-ibo': 'scripture_cc/no semdoms available/ibo-ibo.txt',
        # 'bid-rus': 'scripture_public_domain/no semdoms available/rus-russyn.txt',
        # 'bid-vie': 'scripture_public_domain/no semdoms available/vie-vie1934.txt',
        # 'bid-swp': 'scripture_public_domain/no semdoms available/swp-swp.txt',

        # 'bid-eng-asvbt': 'scripture_public_domain/eng-engasvbt.txt',
        # 'bid-eng-asv': 'scripture_public_domain/eng-eng-asv.txt',
        # 'bid-eng-BBE': 'scripture_public_domain/eng-engBBE.txt',
        # 'bid-eng-Brenton': 'scripture_public_domain/eng-eng-Brenton.txt',
        # 'bid-eng-DBY': 'scripture_public_domain/eng-engDBY.txt',
        # 'bid-eng-DRA': 'scripture_public_domain/eng-engDRA.txt',
        # 'bid-eng-gnv': 'scripture_public_domain/eng-enggnv.txt',
        # 'bid-eng-jps': 'scripture_public_domain/eng-engjps.txt',
        # 'bid-eng-kjv2006': 'scripture_public_domain/eng-eng-kjv2006.txt',
        # 'bid-eng-kjvcpb': 'scripture_public_domain/eng-engkjvcpb.txt',
        # 'bid-eng-kjv': 'scripture_public_domain/eng-eng-kjv.txt',
        # 'bid-eng-lee': 'scripture_public_domain/eng-englee.txt',
        # 'bid-eng-lxx2012': 'scripture_public_domain/eng-eng-lxx2012.txt',
        # 'bid-eng-lxxup': 'scripture_public_domain/eng-englxxup.txt',
        # 'bid-eng-noy': 'scripture_public_domain/eng-engnoy.txt',
        # 'bid-eng-oebcw': 'scripture_public_domain/eng-engoebcw.txt',
        # 'bid-eng-oebus': 'scripture_public_domain/eng-engoebus.txt',
        # 'bid-eng-oke': 'scripture_public_domain/eng-engoke.txt',
        # 'bid-eng-rv': 'scripture_public_domain/eng-eng-rv.txt',
        # 'bid-eng-tnt': 'scripture_public_domain/eng-engtnt.txt',
        # 'bid-eng-uk-lxx2012': 'scripture_public_domain/eng-eng-uk-lxx2012.txt',
        # 'bid-eng-webbe': 'scripture_public_domain/eng-eng-webbe.txt',
        # 'bid-eng-web-c': 'scripture_public_domain/eng-eng-web-c.txt',
        # 'bid-eng-webpb': 'scripture_public_domain/eng-engwebpb.txt',
        # 'bid-eng-webp': 'scripture_public_domain/eng-engwebp.txt',
        # 'bid-eng-webster': 'scripture_public_domain/eng-engwebster.txt',
        # 'bid-eng-web': 'eBible/eng-eng-web.txt',
        # 'bid-eng-wmbb': 'scripture_public_domain/eng-engwmbb.txt',
        # 'bid-eng-wmb': 'scripture_public_domain/eng-engwmb.txt',
        # 'bid-eng-Wycliffe': 'scripture_public_domain/eng-engWycliffe.txt',
        # 'bid-eng-ylt': 'scripture_public_domain/eng-engylt.txt',
        # 'bid-eng-niv11': 'scripture_public_domain/extra_english_bibles/en-NIV11.txt',
        # 'bid-eng-niv84': 'scripture_public_domain/extra_english_bibles/en-NIV84.txt',
        # 'bid-eng-REB89': 'scripture_public_domain/extra_english_bibles/en-REB89.txt',  # mentions "Euphrates" 65 times
        #
        # 'bid-fra-fob': 'eBible/fra-fra_fob.txt',
        # 'bid-fra-lsg': 'scripture_public_domain/fra-fraLSG.txt',
        #
        # 'bid-spa': 'eBible/spa-spaRV1909.txt',
        # 'bid-ind': 'eBible/ind-ind.txt',
        # 'bid-tel': 'scripture_public_domain/tel-telirv.txt',
        # 'bid-tha': 'eBible/tha-thaKJV.txt',
        # 'bid-hin': 'eBible/hin-hin2017.txt',
        # 'bid-nep': 'eBible/npi-npiulb.txt',
        # 'bid-urd': 'scripture_public_domain/urd-urdgvu.txt',
        # 'bid-por': 'eBible/por-porbrbsl.txt',
        # 'bid-swa': 'eBible/swh-swh1850.txt',
        # 'bid-mya': 'eBible/mya-mya.txt',
        # 'bid-arb': 'scripture_public_domain/arb-arb-vd.txt',
        # 'bid-azb': 'scripture_cc/azb-azb.txt',
        #
        # 'bid-deu': 'eBible/no semdoms available/deu-deuelo.txt',
        # 'bid-rus': 'scripture_public_domain/no semdoms available/rus-russyn.txt',
        # 'bid-vie': 'scripture_public_domain/no semdoms available/vie-vie1934.txt',
        # 'bid-tpi': 'eBible/no semdoms available/tpi-tpipng.txt',  # mentions "Yufretis" 65 times
        # 'bid-swp': 'scripture_public_domain/no semdoms available/swp-swp.txt',
        # 'bid-cmn': 'eBible/no semdoms available/cmn-cmn-cu89s.txt',
        # 'bid-yor': 'eBible/no semdoms available/yor-yor2017.txt',
        # 'bid-meu': 'eBible/no semdoms available/meu-meu.txt',
        # 'bid-meu-hmo': 'scripture_cc/no semdoms available/meu-meu-hmo.txt',
        # 'bid-mal': 'eBible/mal-mal.txt',
        # 'bid-pes': 'scripture_cc/no semdoms available/pes-pes.txt',
        # 'bid-gej': 'scripture_copyrighted/no semdoms available/gej-GEN.txt',
    }

    def __init__(self, bids, score_threshold=0.5,
                 state_files_path='data/0_state',
                 aligned_bibles_path='data/1_aligned_bibles',
                 sd_path_prefix='data/4_semdoms/semdom_qa_clean',
                 gt_langs=None):
        assert len(bids) == len(set(bids))  # unique elements
        assert len(self.BIBLES_BY_BID.values()) == len(set(self.BIBLES_BY_BID.values()))  # unique values

        self.bids = bids
        self.bibles_by_bid = {bid: DictionaryCreator.BIBLES_BY_BID[bid] for bid in bids}
        self.source_lang = self._convert_bid_to_lang(self.bids[0])
        self.target_langs = sorted(set([self._convert_bid_to_lang(bid) for bid in self.bids]))
        if gt_langs is None:
            gt_langs = self.target_langs
        else:
            assert set(gt_langs).issubset(self.target_langs)
        self.gt_langs = gt_langs
        self.state_files_base_path = state_files_path
        self.aligned_bibles_path = aligned_bibles_path
        self.tokenizer_by_lang = defaultdict(lambda: 'pre-tokenized')
        self.tokenizer_by_lang.update({lang: 'Stanza' for lang in self.STANZA_LANG_MAP.keys()})
        agglutinative_languages = ['mal', 'npi', 'swh', 'ben', 'ibo', 'azb', 'tha']
        assert len(set(agglutinative_languages).intersection(set(self.STANZA_LANG_MAP.keys()))) == 0
        self.tokenizer_by_lang.update({lang: 'SentencePiece' for lang in agglutinative_languages})
        self.lemmatizer_by_lang = defaultdict(lambda: 'no-stemming')  # heuristic-stemmer')
        self.lemmatizer_by_lang.update({lang: 'Stanza' for lang in self.STANZA_LANG_MAP.keys()})
        self.normalizer = 'lowercase'  # 'lowercase-and-nfkd'
        # self.eng_lemmatizer = WordNetLemmatizer()
        self.suffixes_by_lang = dict()
        self.state_loaded = False
        self.score_threshold = score_threshold
        self.sd_path_prefix = sd_path_prefix
        self.output_file = 'data/evaluation_results.md'
        self.write_buffer = ''
        self.start_timestamp = datetime.fromtimestamp(time.time_ns() / 1e9).strftime('%Y-%m-%d %H:%M:%S')
        self.num_verses = 41899
        self.saved_variables = {'progress_log', 'sds_by_lang', 'verses_by_bid', 'words_by_text_by_lang',
                                'question_by_qid_by_lang', 'wtxts_by_verse_by_bid',
                                'aligned_wtxts_by_qid_by_lang_by_lang', 'aligned_wtxts_by_qid_by_lang_by_lang',
                                'top_scores_by_qid_by_lang',
                                'evaluation_results_by_lang'}

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
        with open('data/verse_ids.txt', 'r') as f:
            self.verse_ids = f.read().splitlines()

        # Saved data (training)
        self.top_scores_by_qid_by_lang = defaultdict(dict)

        # Saved data (evaluation)
        self.evaluation_results_by_lang = defaultdict(dict)

    @staticmethod
    def _convert_bid_to_lang(bid):
        return bid[4:7]

    def _convert_lang_to_bid(self, lang):
        # look for bid that starts with lang in self.bids
        bids = [bid for bid in self.bids if self._convert_bid_to_lang(bid) == lang]
        assert len(bids) == 1
        return bids[0]

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
        elif word.iso_language == 'gej':
            return unidecode(word.display_text)
        return word.display_text

    @staticmethod
    def _apply_prediction(func, ebunch):
        """Applies the given function to each edge in the specified iterable
        of edges.
        """
        return ((u, v, func(u, v)) for u, v in ebunch)

    def _save_state(self):
        state_files_directory = self.start_timestamp + ' ' + ' '.join(self.bids)
        assert ':' in state_files_directory
        print(f'Progress: {self.progress_log}')
        print(f'\nSaving state {state_files_directory}...')
        # create directory if it doesn't exist
        state_files_directory = os.path.join(self.state_files_base_path, state_files_directory)
        if not os.path.exists(state_files_directory):
            os.makedirs(state_files_directory)

        for variable_name in self.saved_variables:
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

        print('State saved.\n')
        sys.stdout.flush()

    def _find_state_directory(self):
        """
        Directory names start with timestamps. Find the most recent directory that contains all the used BIDs.
        """
        directories = os.listdir(self.state_files_base_path)
        for bid in self.bids:
            directories = [directory for directory in directories if bid in directory]
        directories = [directory for directory in directories if directory.count('bid') == len(self.bids)]
        directories.sort()

        most_recent_directory = None
        if len(directories):
            most_recent_directory = directories[-1]

            # This dc should be newer than any other dc, and we do not need to load the own state.
            most_recent_timestamp = ' '.join(most_recent_directory.split(' ')[:2])
            assert most_recent_timestamp <= self.start_timestamp
            self.start_timestamp = most_recent_timestamp
        return most_recent_directory

    def _load_state(self):
        # file path format: {state_files_base_path}/{start_timestamp} {bids}/{variable_name}_{key}.dill
        if self.state_loaded:
            return

        state_files_directory = self._find_state_directory()
        if state_files_directory is None:
            print('Skipped loading because no state files could be found.')
            self.state_loaded = True
            return

        print(f'Loading state {state_files_directory}...')
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
                except (EOFError, TypeError, UnpicklingError):
                    print(f'- WARNING: {path} is broken. Skipping.')
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
        # load sds and Bible verses for all languages
        languages = set([self._convert_bid_to_lang(bid) for bid in self.bids])
        for lang in tqdm(languages, desc='Loading semantic domains', total=len(languages)):
            # load sds
            if lang == 'eng':
                # also load questions without answers to have a complete list of questions
                sd_path = f'{self.sd_path_prefix.replace("_clean", "")}_{lang}.csv'
            else:
                sd_path = f'{self.sd_path_prefix}_{lang}.csv'
            if os.path.isfile(sd_path):
                self.sds_by_lang[lang] = pd.read_csv(sd_path)
            else:
                print(f'WARNING: Unable to load {sd_path}')
                # create empty dataframe
                self.sds_by_lang[lang] = pd.DataFrame(
                    {'cid': [], 'category': [], 'question_index': [], 'question': [], 'answer': []})
                if lang not in ('deu', 'rus', 'vie', 'tpi', 'swp', 'meu', 'gej', 'yor', 'hmo', 'azb', 'ibo'):
                    raise FileNotFoundError(f'Unable to load {sd_path}')

        for bid in tqdm(self.bids, desc='Loading Bibles', total=len(self.bids)):
            # load Bible verses
            with open(os.path.join('data/5_raw_bibles', self.bibles_by_bid[bid]),
                      'r') as bible:
                self.verses_by_bid[bid] = bible.readlines()
            # self.verses_by_bid[bid] = [verse if verse != '<range>\n' else '\n' for verse in self.verses_by_bid[bid]]
            assert len(self.verses_by_bid[bid]) == self.num_verses

    def _load_additional_word_qid_mappings(self):
        path = 'data/4_semdoms/additional_words.xlsx'
        df = pd.read_excel(path)
        for index, row in tqdm(df.iterrows(), desc=f'Loading additional word-QID mappings from {path}', total=len(df)):
            wtxt = str(row.word)
            lang = row.language
            if lang not in self.words_by_text_by_lang or wtxt not in self.words_by_text_by_lang[lang]:
                continue
            qid = row.qid
            gpt_4_answer = row['GPT-4 answer']
            if gpt_4_answer == 1:
                self.words_by_text_by_lang[lang][wtxt].qids.add(qid)
            if gpt_4_answer == 0 \
                    and wtxt in self.words_by_text_by_lang[lang] \
                    and qid in self.words_by_text_by_lang[lang][wtxt].qids:
                print(lang, wtxt, qid, row['source model'])
                # Ground-truth words can only be forbidden manually
                if not (row['source model'][:6] == 'manual' or row['GPT-4 original answer'] == 1) \
                        and row['manually verified'] == 1:
                    print(f'Unexpected GPT-4 answer for ground-truth word {lang} {wtxt} {qid}: {gpt_4_answer}')
                self.words_by_text_by_lang[lang][wtxt].qids.discard(qid)

    def split_sd_answers(self, row):
        try:
            question = row.question.replace("'", '')
            question = question.replace('"', '')
            answer = row.answer.replace("'", '') if row.answer == row.answer else ''
            answer = answer.replace('"', '')
        except AttributeError:
            print(row)
            raise
        assert int(row.question_index) >= 0  # assert that question_index is parsable
        qid = f'{row.cid} {row.question_index}'
        wtxts = {wtxt for wtxt in re.split(r'[,、،၊]', answer) if wtxt}  # split by different commas

        # handle parentheses
        words_without_parentheses = set()
        for i, wtxt in enumerate(wtxts):
            text_in_parentheses = re.findall(r'\([^)]*\)', wtxt)
            if len(text_in_parentheses) == 0:
                words_without_parentheses.add(wtxt)
                continue
            text_in_parentheses = text_in_parentheses[0]  # We rarely have more than one parentheses block.

            if text_in_parentheses[-2:] != '.)' and text_in_parentheses not in ('(n)', '(v)', '(adj)', '(adv)'):
                # remove parentheses
                new_word = wtxt.replace('(', '').replace(')', '')
                words_without_parentheses.add(new_word)

            # add text without text in parentheses
            new_word = re.sub(r'\([^)]*\)', '', wtxt)
            new_word = re.sub(r' +', ' ', new_word)  # replace double spaces with single spaces
            words_without_parentheses.add(new_word)

        wtxts = [wtxt.strip() for wtxt in words_without_parentheses]

        # remove empty words
        wtxts = [wtxt for wtxt in wtxts if wtxt.strip()]

        return qid, question, wtxts

    def _build_sds(self):
        """
        Convert a semantic domain dataframe to a dictionary.
        :return: None
        """
        words_by_lang = defaultdict(list)

        for lang, sds in tqdm(self.sds_by_lang.items(), desc='Building semantic domains', total=len(self.sds_by_lang)):
            self.question_by_qid_by_lang[lang] = {}  # show that we tried building sds for this language
            if self.sds_by_lang[lang].empty:
                continue
            for index, row in sds.iterrows():
                # if row.cid.startswith('9'):
                #     continue
                qid, question, wtxts = self.split_sd_answers(row)
                # if lang == 'eng':
                #     wtxts = self._lemmatize_english_verse(wtxts)
                words_by_lang[lang].append((qid, question, wtxts))

            qids, questions, wtxts = zip(*words_by_lang[lang])
            if lang in words_by_lang and self.lemmatizer_by_lang[lang] == 'Stanza':
                wtxts = self._run_stanza_pipeline(wtxts, lang, is_tokenized=True)
            elif self.lemmatizer_by_lang[lang] == 'heuristic-stemmer':
                wtxts = self._stem_lines(wtxts, lang)
            else:
                # no lemmatization/stemming
                pass
            wtxts = self._normalize_words(wtxts)
            words_by_lang[lang] = list(zip(qids, questions, wtxts))

        for lang in words_by_lang:
            for qid, question, wtxts in words_by_lang[lang]:
                words = {wtxt: Word(wtxt, lang, {qid}) for wtxt in wtxts}

                # add new words to words_by_text_by_lang
                for word in words.values():
                    if word.text not in self.words_by_text_by_lang[lang]:
                        self.words_by_text_by_lang[lang][word.text] = word
                    else:
                        self.words_by_text_by_lang[lang][word.text].qids.add(qid)

                self.question_by_qid_by_lang[lang][qid] = question

    def _stem_word(self, word, lang):
        suffixes = self.suffixes_by_lang[lang]

        # remove as many suffixes as possible
        # remove the longest possible suffixes first
        # single_character_removed = False
        removed_suffixes = []
        while len(word) > 1:
            longest_matching_suffix = None
            for suffix in suffixes:
                if word.endswith(suffix):
                    if longest_matching_suffix is None or len(suffix) > len(longest_matching_suffix):
                        longest_matching_suffix = suffix

            if longest_matching_suffix is not None:
                # only remove the suffix if the stemmed word occurs in the trie
                # if len(longest_matching_suffix) == 1:
                # if single_character_removed:
                #     break
                # else:
                #     single_character_removed = True
                stemmed_word = word[:-len(longest_matching_suffix)]
                word = stemmed_word
                removed_suffixes.append(longest_matching_suffix)
            else:
                break

        return word, removed_suffixes[::-1]

    def _train_stemmer(self, wtxts_by_line, lang):
        # flatten wtxts_by_line
        all_wtxts = set()
        for wtxts in wtxts_by_line:
            all_wtxts.update(wtxts)

        # build a trie and count the number of occurrences of each edge
        stem_trie = {}
        for word in all_wtxts:
            node = stem_trie
            for i in range(len(word)):
                if word[i] not in node:
                    node[word[i]] = {}
                node = node[word[i]]
                if 'count' not in node:
                    node['count'] = 0
                node['count'] += 1

        # build a suffix trie and count the number of occurrences of each edge (just like above with the word backwards)
        suffix_trie = {}
        for word in all_wtxts:
            node = suffix_trie
            for i in range(len(word) - 1, -1, -1):
                if word[i] not in node:
                    node[word[i]] = {}
                node = node[word[i]]
                if 'count' not in node:
                    node['count'] = 0
                node['count'] += 1
        suffix_trie['count'] = len(all_wtxts)

        def find_paths(node, current_path=None, threshold=None):
            paths = []

            if current_path is None:
                current_path = ''

            if threshold is None:
                threshold = node['count']

            for char, child_node in node.items():
                if char != 'count' and (child_node['count'] >= threshold / 3 or current_path == ''):
                    paths.extend(find_paths(child_node, char + current_path, child_node['count']))

            return paths + [current_path]

        # Using the function on the constructed suffix_trie
        deep_suffixes = find_paths(suffix_trie)

        # Identify the most common suffixes
        suffixes = []
        for word in all_wtxts:
            for i in range(1, len(word)):
                suffixes.append(word[i:])
        suffix_counts = Counter(suffixes)

        suffix_threshold = len(all_wtxts) / 25
        deep_suffix_threshold = len(all_wtxts) / 250

        # filter all suffix counts that occur less than ... times
        suffix_counts = Counter({suffix: count for suffix, count in suffix_counts.items() if
                                 count > suffix_threshold or (
                                         suffix in deep_suffixes and count > deep_suffix_threshold and len(
                                     suffix) > 1)})

        # If a suffix A (e.g., -ng) is a suffix of suffix B (e.g., -ing), subtract the count of B from the count of A
        # Example:
        #   suffix_counts = {'g': 100, 'ng': 50, 'ing': 30}
        #   suffix_counts_cleaned = {'g': 50, 'ng': 20, 'ing': 30}
        # Use a two-pointer algorithm to avoid quadratic runtime
        # Sort suffixes by length
        # start with the shortest suffix
        # only compare suffixes if one is exactly one character longer than the other
        suffixes = sorted(suffix_counts.keys(), key=len)
        suffix_counts_cleaned = suffix_counts.copy()
        for short_suffix_idx, short_suffix in enumerate(suffixes):
            for long_suffix_idx in range(short_suffix_idx + 1, len(suffixes)):
                long_suffix = suffixes[long_suffix_idx]
                if len(long_suffix) - len(short_suffix) > 1:
                    break
                if long_suffix[1:] == short_suffix:
                    suffix_counts_cleaned[short_suffix] -= suffix_counts[long_suffix]

        # make suffix_counts_cleaned a sorted list of tuples
        suffix_counts_cleaned = sorted(suffix_counts_cleaned.items(), key=lambda x: x[1], reverse=True)

        suffix_counts_cleaned = [(suffix, count) for suffix, count in suffix_counts_cleaned if
                                 count >= suffix_threshold or (
                                         suffix in deep_suffixes and count > deep_suffix_threshold and len(suffix) > 1)]

        self.suffixes_by_lang[lang] = [suffix for suffix, count in suffix_counts_cleaned]

    def _normalize_words(self, wtxts_by_line):
        return [[self._normalize_word(wtxt) for wtxt in line] for line in wtxts_by_line]

    def _normalize_word(self, wtxt):
        wtxt = wtxt.strip()
        if 'lowercase' in self.normalizer:
            wtxt = wtxt.lower()
        if 'nfkd' in self.normalizer:
            wtxt = self._remove_diacritics(wtxt)
        return wtxt

    def _remove_diacritics(self, input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _stem_lines(self, wtxts_by_line, lang, train_stemmer=False):
        assert self.lemmatizer_by_lang[lang] == 'heuristic-stemmer'

        print(f'Stemming {lang}...')

        wtxts_by_line = self._normalize_words(wtxts_by_line)

        if train_stemmer:
            self._train_stemmer(wtxts_by_line, lang)
        assert lang in self.suffixes_by_lang, f'Stemmer for {lang} has not been trained yet.'

        stemmed_lines = []
        for line in tqdm(wtxts_by_line, desc=f'Stemming {lang}...', total=len(wtxts_by_line)):
            stemmed_line = []
            for wtxt in line:
                stem, removed_suffixes = self._stem_word(wtxt, lang)
                # if wtxt != stem:
                #     stem += '|' + wtxt
                stemmed_line.append(stem)
            stemmed_lines.append(stemmed_line)

        return stemmed_lines

    def _run_stanza_pipeline(self, lines, lang, is_tokenized, save_bid=None, batch_size=None):
        assert self.lemmatizer_by_lang[lang] == 'Stanza'

        print(f'Running Stanza pipeline for {lang}')
        # Tokenization, multi-word token expansion, POS tagging, lemmatization

        if batch_size is None:
            batch_size = ceil(self.num_verses / 10)

        processors = 'tokenize,mwt,pos,lemma' if lang not in ('hin', 'cmn', 'tel', 'urd') else 'tokenize,pos,lemma'
        if save_bid is not None:
            file_path = os.path.join('data/10_lemmatized_bibles', f'{save_bid}_{processors}.dill')
            if os.path.isfile(file_path):
                print(f'{file_path} already exists. Loading...')
                with open(file_path, 'rb') as f:
                    return dill.load(f)

        delimiter = ' _ # '  # two delimiters are more accurate than one
        empty_lines = []
        docs = []

        for i, line in enumerate(lines):
            if (type(line) == list and len(line) == 0) \
                    or (type(line) == str and len(line.strip()) == 0):
                # do not lemmatize empty lines to avoid crashes
                empty_lines.append(i)
        assert len(empty_lines) < len(lines)

        if is_tokenized:
            # convert each line from ['word1', 'word2', ...] to [{'id': 1, 'text': 'word1'}, {'id': 2, 'text': 'word2'}, ...]
            tokenized_lines = []
            for i, line in enumerate(lines):
                if i in empty_lines:
                    continue
                tokenized_lines.append([{'id': j, 'text': word} for j, word in enumerate(line, start=1)])
            docs.append(stanza.Document(tokenized_lines))
        else:
            assert self.tokenizer_by_lang[lang] == 'Stanza'
            for line in lines:
                assert '_' not in line and '#' not in line, f'Forbidden string in {line}: {delimiter.strip()}'
            doc_lines = []
            for i, line in enumerate(lines):
                if i in empty_lines:
                    continue
                doc_lines.append(line)
                if len(doc_lines) == batch_size:
                    docs.append(delimiter.join(doc_lines))
                    doc_lines = []
            if len(doc_lines):
                docs.append(delimiter.join(doc_lines))

        # print(tokenized_lines[:5])
        # print(f'Empty lines: {len(empty_lines)}')
        print(f'{lang} document created.')

        lang_code = self.STANZA_LANG_MAP[lang]
        stanza.download(lang_code)
        nlp = stanza.Pipeline(lang=lang_code, processors=processors, allow_unknown_language=False,
                              tokenize_pretokenized=is_tokenized)

        # func = partial(process_text, lang_code, processors, is_tokenized)
        # num_processes = 1 # min(2, mp.cpu_count() // 2)
        # with mp.Pool(processes=num_processes) as pool:
        #     docs = tqdm(pool.map(func, docs), desc=f'Running Stanza pipeline for {lang}', total=len(docs))

        tokenized_sentences = []
        for doc in tqdm(docs, desc=f'Running Stanza pipeline for {lang}', total=len(docs)):
            doc = nlp(doc)  # already parallelized
            if is_tokenized:
                tokenized_sentences.extend([sentence.words for sentence in doc.sentences])
            else:
                # Split tokenized output based on the delimiter
                current_sentence = []
                for sentence in doc.sentences:
                    for word in sentence.words:
                        if '_' in word.text:
                            assert word.text.count('_') == 1
                            tokenized_sentences.append(current_sentence)
                            current_sentence = []
                        elif '#' in word.text:
                            pass  # ignore additional delimiter
                        else:
                            current_sentence.append(word)
                tokenized_sentences.append(current_sentence)
            assert len(tokenized_sentences) % batch_size == 0 or len(tokenized_sentences) == len(
                lines) - len(
                empty_lines), f'Length mismatch: {len(tokenized_sentences)} % {batch_size} != 0 and {len(tokenized_sentences)} != {len(lines) - len(empty_lines)}'

        lemmas = [[(word.lemma if word.lemma else word.text) for word in sentence] for sentence in tokenized_sentences]

        # set lemmas = [] for each empty line
        for i in empty_lines:
            lemmas.insert(i, [])

        assert len(lemmas) == len(lines), f'Length mismatch: {len(lemmas)} != {len(lines)}'

        # assert that only the lines in empty_lines are empty
        for i, line in enumerate(lemmas):
            if i in empty_lines:
                assert len(line) == 0
            else:
                assert len(line) > 0

        if save_bid is not None:
            # save lemmas
            print(f'Saving {file_path}...')
            with open(file_path, 'wb') as f:
                dill.dump(lemmas, f)

        return lemmas

    def _lemmatize_english_verse(self, verse):
        for wtxt in verse:
            assert '_' not in wtxt, f'Unexpected underscore in {wtxt}'

        # https://stackoverflow.com/a/57686805/8816968
        lemmatized_wtxts = []
        pos_labels = pos_tag(verse)
        for wtxt, pos_label in pos_labels:
            if wtxt in ('as', 'us'):
                # do not replace 'as' with 'a' or 'us' with 'u'
                lemmatized_wtxts.append(wtxt)
                continue

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

        ## formerly needed by SemanticDomainIdentifier
        ## replace each word that changed with "original_lemma"
        # for i, (wtxt, lemmatized_wtxt) in enumerate(zip(verse, lemmatized_wtxts)):
        #     if wtxt != lemmatized_wtxt:
        #         lemmatized_wtxts[i] = f'{wtxt}_{lemmatized_wtxt}'

        return lemmatized_wtxts

    def _tokenize_verses(self):
        for bid in tqdm(self.bids, desc='Tokenizing verses', total=len(self.bids)):
            lang = self._convert_bid_to_lang(bid)

            if lang and self.tokenizer_by_lang[lang] == 'SentencePiece':
                assert self.lemmatizer_by_lang[lang] == 'no-stemming'
                file = os.path.join('data/5_raw_bibles', self.bibles_by_bid[bid])
                vocab_size = 10000 if bid != 'bid-swh-1850' else 5000  # swh Bible is too small for 10k
                SentencePieceTrainer.train(input=file, model_prefix=bid, vocab_size=vocab_size)
                sp = SentencePieceProcessor()
                sp.load(f'{bid}.model')
                wtxts_by_verse = [[token for token in sp.encode_as_pieces(verse)] for verse
                                  in self.verses_by_bid[bid]]
            elif self.tokenizer_by_lang[lang] == 'Stanza':
                wtxts_by_verse = self._run_stanza_pipeline(self.verses_by_bid[bid], lang, is_tokenized=False,
                                                           save_bid=bid)
            else:
                assert self.tokenizer_by_lang[lang] == 'pre-tokenized'
                tokenizer = Tokenizer(BPE())  # dummy tokenizer
                tokenizer.pre_tokenizer = BertPreTokenizer()

                # trainer = BpeTrainer(
                #     # min_frequency=1,
                #     show_progress=False
                # )
                # tokenizer.train(files=[file], trainer=trainer)

                # tokenize all verses
                # wtxts_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_bid[bid]]
                wtxts_by_verse = [[token for token, idxs in tokenizer.pre_tokenizer.pre_tokenize_str(verse)] for verse
                                  in self.verses_by_bid[bid]]

                if self.lemmatizer_by_lang[lang] == 'heuristic-stemmer':
                    wtxts_by_verse = self._stem_lines(wtxts_by_verse, lang, train_stemmer=True)
            wtxts_by_verse = self._normalize_words(wtxts_by_verse)

            # clean up all wtxts
            wtxts_by_verse = [[wtxt.replace('▁', '') for wtxt in verse] for verse in wtxts_by_verse]

            # remove empty words
            wtxts_by_verse = [[wtxt for wtxt in verse if wtxt] for verse in wtxts_by_verse]

            ## lemmatize all English words
            # if lang == 'eng':
            #     wtxts_by_verse = [self._lemmatize_english_verse(verse) for verse in
            #                       tqdm(wtxts_by_verse, total=len(wtxts_by_verse),
            #                            desc='Lemmatizing English verses')]

            self.wtxts_by_verse_by_bid[bid] = wtxts_by_verse.copy()

            # mark words as appearing in the Bible
            wtxts = [wtxt for wtxts in wtxts_by_verse for wtxt in wtxts]
            for wtxt in wtxts:
                if wtxt in self.words_by_text_by_lang[lang]:
                    self.words_by_text_by_lang[lang][wtxt].occurrences_in_bible += 1
                else:
                    self.words_by_text_by_lang[lang][wtxt] = Word(wtxt, lang, set(), 1)

    def _create_aligned_bibles_file_base_name(self, bid_1, bid_2):
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)
        tokenizer_1 = self.tokenizer_by_lang[lang_1]
        tokenizer_2 = self.tokenizer_by_lang[lang_2]
        lemmatizer_1 = self.lemmatizer_by_lang[lang_1]
        lemmatizer_2 = self.lemmatizer_by_lang[lang_2]
        # return f'{bid_1}_tok-{tokenizer_1}_lem-{lemmatizer_1}_{bid_2}_tok-{tokenizer_2}_lem-{lemmatizer_2}'
        return f'nor-{self.normalizer}_{bid_1}_tok-{tokenizer_1}_lem-{lemmatizer_1}_{bid_2}_tok-{tokenizer_2}_lem-{lemmatizer_2}'

    def _combine_alignments(self):
        # combine verses from two different Bibles into a single file for wtxt aligner (Eflomal)
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                lang_1 = self._convert_bid_to_lang(bid_1)
                lang_2 = self._convert_bid_to_lang(bid_2)
                if not (lang_1 == lang_2 and lang_1 == 'eng') and (
                        bid_1 >= bid_2 or not (lang_1 in self.gt_langs or lang_2 in self.gt_langs)):
                    # map every pair of different Bibles (and eng-eng, for stopword removal), as long as one of them has known semantic domains
                    continue

                aligned_bibles_file_base_name = self._create_aligned_bibles_file_base_name(bid_1, bid_2)
                aligned_bibles_file_path = f'{self.aligned_bibles_path}/{aligned_bibles_file_base_name}_eflomal_diag.align'
                if os.path.isfile(aligned_bibles_file_path):
                    print(f'Skipped: Aligned Bibles file {aligned_bibles_file_path} already exists')
                    continue

                combined_bibles_file_path = f'{self.aligned_bibles_path}/{aligned_bibles_file_base_name}.txt'
                with open(combined_bibles_file_path, 'w') as combined_bibles:
                    for idx, (bid_1_wtxts, bid_2_wtxts) in tqdm(enumerate(
                            zip(self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2])),
                            desc=f'Combining alignments for {bid_1} and {bid_2}',
                            total=len(self.wtxts_by_verse_by_bid[bid_1])):
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0 and len(bid_1_wtxts) + len(bid_2_wtxts) > 0:
                            # verse is missing in only one bible
                            bible_reference = src.utils.convert_verse_id_to_bible_reference(idx)
                            print('Missing verse - verses might be misaligned! verse idx:', idx, bible_reference, bid_1, bid_2,
                                  bid_1_wtxts, bid_2_wtxts)
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0:
                            # verse is missing in both Bibles
                            bid_1_wtxts = ['#placeholder#']
                            bid_2_wtxts = ['#placeholder#']
                        combined_bibles.write(' '.join(bid_1_wtxts) + ' ||| ' + ' '.join(bid_2_wtxts) + '\n')

                result = subprocess.run(
                    ['sh', 'align_bibles.sh', f'{self.aligned_bibles_path}/{aligned_bibles_file_base_name}'],
                    capture_output=True, text=True)
                print(result.stderr)
                ## retrieve the final entropy and perplexity
                # matches = re.search(r'FINAL(.|\n)*cross entropy: (\d+\.\d+)\n *perplexity: (\d+\.\d+)', result.stderr)
                # cross_entropy = float(matches.group(2))
                # perplexity = float(matches.group(3))
                # print(f'cross entropy: {cross_entropy}, perplexity: {perplexity}')

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
                    assert len(loaded_variable) == len(
                        self.target_langs), f'Loaded variable {loaded_variable_name} has wrong length ({len(loaded_variable)} != {len(self.target_langs)})'

        return target_progress in self.progress_log

    def _set_progress(self, progress, save):
        self.progress_log.append(progress)
        if save:
            self._save_state()

    def _execute_and_track_state(self, func, step_name=None, save=False, load=False, *args, **kwargs):
        if step_name is None:
            step_name = func.__name__
        if self._check_already_done(step_name, load) and step_name != '_evaluate':
            print(f'Skipped: {step_name} already done')
            return
        print(f'Executing: {step_name}')
        func(*args, **kwargs)
        self._set_progress(step_name, save)

    def _preprocess_data(self):
        self._load_data()
        self._tokenize_verses()
        self._build_sds()
        # self._load_additional_word_qid_mappings()
        self._combine_alignments()

    def _add_bidirectional_edge(self, word_1, word_2, count=1):
        word_1.add_aligned_word(word_2, count)
        word_2.add_aligned_word(word_1, count)

    def _map_word_to_qids(self, source_wtxt, target_wtxt, source_lang, target_lang):
        """
        Assign a target word to all of a source word's qids.
        """
        for new_qid in self.words_by_text_by_lang[source_lang][source_wtxt].qids:
            self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][source_lang][new_qid] += ', ' + target_wtxt

    def _map_word_to_qid_bidirectionally(self, wtxt_1, wtxt_2, lang_1, lang_2):
        """
        Assign two words to all of each other's qids.
        """
        self._map_word_to_qids(wtxt_1, wtxt_2, lang_1, lang_2)
        if lang_1 == lang_2 and wtxt_1 == wtxt_2:
            return
        self._map_word_to_qids(wtxt_2, wtxt_1, lang_2, lang_1)

    def _map_two_bibles_bidirectionally(self, alignment, bid_1, bid_2):
        # map words in two Bibles to semantic domains
        # Caveat: This function ignores wtxts that could not be aligned.
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)

        # at least create an empty dictionary to show that we tried aligning words
        assert (lang_2 not in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1])
        assert (lang_1 not in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2])
        self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1][lang_2] = defaultdict(str)
        self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2][lang_1] = defaultdict(str)

        for idx, (alignment_line, wtxts_1, wtxts_2) in tqdm(
                enumerate(zip(alignment, self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2])),
                desc=f'Map {bid_1} and {bid_2} words and semantic domain questions bidirectionally',
                total=len(self.verses_by_bid[bid_1])):
            if alignment_line == '\n':
                assert (len(wtxts_1) * len(wtxts_2) == 0
                        or len(wtxts_1) + len(wtxts_2) > 100)  # word aligner skipped this verse because it is too long
                continue
            if alignment_line == '0-0\n' and len(wtxts_1) * len(wtxts_2) == 0:
                continue
            alignment_line = alignment_line.replace('\n', '')
            aligned_wtxt_pairs = alignment_line.split(' ')

            for aligned_wtxt_pair in aligned_wtxt_pairs:
                wtxt_1_idx, wtxt_2_idx = [int(num) for num in aligned_wtxt_pair.split('-')]
                try:
                    wtxt_1 = wtxts_1[wtxt_1_idx]
                    wtxt_2 = wtxts_2[wtxt_2_idx]
                except IndexError:
                    pass
                punctuations = ['.', ',', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '«', '»', '“', '”', '…',
                                '"', "'", '-', '`', '´']
                if wtxt_1 in punctuations or wtxt_2 in punctuations:
                    continue
                word_1 = self.words_by_text_by_lang[lang_1][wtxt_1]
                word_2 = self.words_by_text_by_lang[lang_2][wtxt_2]
                self._map_word_to_qid_bidirectionally(wtxt_1, wtxt_2, lang_1, lang_2)
                self._add_bidirectional_edge(word_1, word_2)

    def _map_words_to_qids(self):
        # map words in all target language Bibles to semantic domains
        count = 0
        total = len(self.gt_langs) ** 2 // 2 + len(self.gt_langs) * (
                    len(self.target_langs) - len(self.gt_langs))  # assumes that we have exactly one bid per lang
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                lang_1 = self._convert_bid_to_lang(bid_1)
                lang_2 = self._convert_bid_to_lang(bid_2)
                if not (lang_1 == lang_2 and lang_1 == 'eng') and (
                        bid_1 >= bid_2 or not (lang_1 in self.gt_langs or lang_2 in self.gt_langs)):
                    # map every pair of different Bibles (and eng-eng, for stopword removal), as long as one of them has known semantic domains
                    continue

                aligned_bibles_file_base_name = self._create_aligned_bibles_file_base_name(bid_1, bid_2)
                with open(f'{self.aligned_bibles_path}/{aligned_bibles_file_base_name}_eflomal_diag.align',
                          'r') as alignment_file:
                    alignment = alignment_file.readlines()
                    assert len(alignment) == self.num_verses
                    self._map_two_bibles_bidirectionally(alignment, bid_1, bid_2)
                    count += 1
                    print(f'{count} / ~{total} completed')

    def _remove_stop_words(self):
        stop_words = find_stop_words(self)
        for lang in stop_words:
            for wtxt in stop_words[lang]:
                word = self.words_by_text_by_lang[lang][wtxt]
                aligned_words = list(word.get_aligned_words_and_counts(self.words_by_text_by_lang))
                for w, _ in aligned_words:
                    word.remove_alignment(w)
                    w.remove_alignment(word)
        # self.remove_stop_words_from_alignment_file(stop_words)

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
        self._print_and_write_metric('precision', precision, f'{num_true_positive_wtxts} '
                                                             f'/ {num_positive_wtxts} found {target_lang} semantic domain words are correct')

        # How many of the target sd wtxts in the ground-truth set were actually found?
        recall = num_true_positive_wtxts / num_total_gt_target_wtxts
        self._print_and_write_metric('recall', recall, f'{num_true_positive_wtxts} '
                                                       f'/ {num_total_gt_target_wtxts} {target_lang} actual semantic domain words found')

        f1 = 0.0 if precision * recall == 0 else 2 * (precision * recall) / (precision + recall)
        self._print_and_write_metric('F1', f1)

        # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # was actually found?
        recall_adjusted = 0.0 if num_true_positive_wtxts == 0 \
            else num_true_positive_wtxts / num_total_gt_sd_wtxts_in_target_verses
        self._print_and_write_metric('recall*', recall_adjusted, f'{num_true_positive_wtxts} '
                                                                 f'/ {num_total_gt_sd_wtxts_in_target_verses} {target_lang} actual semantic domain words '
                                                                 f'- that also appear in the target verses - found')

        f1_adjusted = 0.0 if precision * recall_adjusted == 0 else 2 * (precision * recall_adjusted) / (
                precision + recall_adjusted)
        self._print_and_write_metric('F1*', f1_adjusted)

        # How many of the gt target wtxts appear in the target verses?
        target_wtxt_coverage = num_total_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        self._print_and_write_metric('Ground truth target word coverage', target_wtxt_coverage,
                                     f'{num_total_gt_sd_wtxts_in_target_verses} '
                                     f'/ {num_total_gt_target_wtxts} {target_lang} actual non-unique semantic domain words '
                                     f'also appear in the target verses')

        # # How many of the source wtxts appear in the source verses?
        # source_wtxt_coverage = num_total_sd_wtxts_in_source_verses / num_total_sd_source_wtxts
        # self._print_and_write_metric(f'Source wtxt coverage: {source_wtxt_coverage:.3f} ({num_total_sd_wtxts_in_source_verses} '
        #       f'/ {len(num_total_sd_source_wtxts)} {self.source_language} actual non-unique semantic domain words '
        #       'also appear in the source verses)')

        # optional todo: consider wtxt groups vs. single wtxts in calculation
        # # How many of the single gt target wtxts appear in the target verses?
        # target_wtxt_coverage = num_total_single_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        # self._print_and_write_metric(f'Ground truth single target wtxt coverage: {target_wtxt_coverage:.3f} '
        #       f'({num_total_single_gt_sd_wtxts_in_target_verses} '
        #       f'/ {num_total_gt_target_wtxts} {self.target_language} actual non-unique semantic domain words '
        #       'also appear in the target verses)')
        #
        # # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # # was actually found?
        # recall_adjusted2 = num_true_positive_wtxts / num_total_single_gt_sd_wtxts_in_target_verses
        # self._print_and_write_metric(f'recall**: {recall_adjusted2:.3f} ({num_true_positive_wtxts} '
        #       f'/ {num_total_single_gt_sd_wtxts_in_target_verses} {self.target_language} '
        #       f'actual single semantic domain words '
        #       '- that also appear in the target verses - found)')
        #
        # f1_adjusted2 = 2 * (precision * recall_adjusted2) / (precision + recall_adjusted2)
        # self._print_and_write_metric(f'F1**: {f1_adjusted2:.3f}')
        return precision, recall, f1, recall_adjusted, f1_adjusted

    def _load_test_data(self, target_lang):
        """
        Load source and corresponding target wtxts from Purdue Team (ground truth data for dictionary creation).
        :param target_lang: The language that should be evaluated.
        :return: A dataframe with the source and target wtxts.
        """
        if target_lang in ('urd', 'gej'):
            print(f'Cannot compute MRR for {target_lang} because no ground truth data is available.')
            return None
        elif target_lang in ('tpi', 'swp'):  # , 'meu')
            df_test = pd.read_csv('data/eng-tpi-meu-swp.csv')
        else:
            df_test = pd.read_csv('data/multilingual_semdom_dictionary.csv')
        df_test = df_test[[f'{self.source_lang}-000.txt', f'{target_lang}-000.txt']]
        df_test.columns = ['source_wtxt', 'target_wtxts']
        df_test = df_test[df_test['target_wtxts'].notna()]

        df_test['source_wtxt'] = df_test['source_wtxt'].apply(lambda x: [self._normalize_word(i) for i in x.split('/')])
        df_test['target_wtxts'] = df_test['target_wtxts'].apply(lambda x: [self._normalize_word(i) for i in x.split('/')])
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
        target_qids = defaultdict(list)
        df_test = self._load_test_data(target_lang)
        if df_test is None:
            return None

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
                print('\t' + str(self.question_by_qid_by_lang[self.source_lang][qid]))
                print('\t found (positive) words: ' + str(wtxt_list))
                print('\t reference (true) words: ' + str(target_wtxts))
                print('\t reciprocal rank:        ' + str(f'{reciprocal_rank:.2f}\n'))
            mean_reciprocal_rank += reciprocal_rank
        mean_reciprocal_rank /= len(target_qids)
        self._print_and_write_metric('MRR', mean_reciprocal_rank,
                                     f'{len(target_qids)} / {len(self.top_scores_by_qid_by_lang[target_lang])} {target_lang} '
                                     f'questions selected')
        return mean_reciprocal_rank

    def _print_and_write(self, text):
        """
        Print text to stdout and a write buffer.
        :return: None
        """
        print(text)
        self.write_buffer += text + '  \n'

    def _print_and_write_metric(self, metric, value, note=''):
        """
        Print formatted metric to stdout and a write buffer.
        """
        self.write_buffer += f'| {metric} | {value:.3f} | {note} |\n'
        padding = ' ' * (10 - len(metric))
        if len(note):
            note = f'({note})'
        print(f'{metric}:{padding} {value:.3f} {note}')

    def _flush(self):
        """
        Flush the write buffer and prepend it to the output file.
        :return: None
        """
        with open(self.output_file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(self.write_buffer + '---\n\n' + content)

    def _evaluate(self, print_reciprocal_ranks=False):
        filtered_target_wtxts_by_qid_by_lang = self._filter_target_sds_with_threshold()
        self._print_and_write(datetime.now().strftime('\n# %d/%m/%Y %H:%M:%S'))
        self._print_and_write(f'Bibles: `{self.bids}`')
        self._print_and_write(f'Threshold: {self.score_threshold}')

        for target_lang in self.target_langs:
            self._print_and_write(f'\n\n## Evaluation for {target_lang}')
            self.write_buffer += '| metric | value | note |\n'
            self.write_buffer += '|:-------|:------|:-----|\n'
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
        self._flush()

    @abstractmethod
    def create_dictionary(self, load=False, save=False, *args, **kwargs):  # pragma: no cover
        pass
