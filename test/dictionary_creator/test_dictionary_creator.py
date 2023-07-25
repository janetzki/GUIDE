import os
import shutil
from unittest import TestCase

from src.dictionary_creator.dictionary_creator import DictionaryCreator
from src.word import Word


class AbstractTestDictionaryCreator(TestCase):
    def setUp(self) -> None:
        self.tested_class = DictionaryCreator  # overwritten by child classes
        self.dc = None  # overwritten by child classes

        # set the working directory to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        os.chdir(project_root)

        # force delete all directories in test/data/0_state
        for directory in os.listdir('test/data/0_state'):
            shutil.rmtree(os.path.join('test/data/0_state', directory))

        # delete all files in test/data/1_aligned_bibles
        for file in os.listdir('test/data/1_aligned_bibles'):
            os.remove(os.path.join('test/data/1_aligned_bibles', file))

        DictionaryCreator.BIBLES_BY_BID.update({
            'bid-eng-DBY-1000': '../../../dictionary_creator/test/data/eng-engDBY-1000-verses.txt',
            'bid-eng-DBY-100': '../../../dictionary_creator/test/data/eng-engDBY-100-verses.txt',
            'bid-eng-DBY-10': '../../../dictionary_creator/test/data/eng-engDBY-10-verses.txt',
            'bid-fra-fob-1000': '../../../dictionary_creator/test/data/fra-fra_fob-1000-verses.txt',
            'bid-fra-fob-100': '../../../dictionary_creator/test/data/fra-fra_fob-100-verses.txt',
            'bid-fra-fob-10': '../../../dictionary_creator/test/data/fra-fra_fob-10-verses.txt',
            'bid-deu-10': '../../../dictionary_creator/test/data/deu-deuelo-10-verses.txt',
        })
        self.maxDiff = 100000

    def _create_dictionary_creator(self, bids=None, sd_path_prefix='test/data/4_semdoms/semdom_qa_clean_short'):
        if bids is None:
            bids = ['bid-eng-DBY-10', 'bid-fra-fob-10']

        dc = self.tested_class(bids, score_threshold=0.2,
                               state_files_path='test/data/0_state',
                               aligned_bibles_path='test/data/1_aligned_bibles',
                               sd_path_prefix=sd_path_prefix)
        dc.num_verses = 10
        return dc

    def _create_word(self, text, lang, qids=None, occurrences_in_bible=1):
        word = Word(text, lang, qids, occurrences_in_bible)
        if self.dc is not None:
            self.dc.words_by_text_by_lang[lang][text] = word
        return word

    def _run_full_pipeline_twice(self, load_1, save_1, load_2, save_2, sd_path_prefix=None, *args, **kwargs):
        dc_new = self._create_dictionary_creator()

        if sd_path_prefix is not None:
            self.dc.sd_path_prefix = sd_path_prefix
            dc_new.sd_path_prefix = sd_path_prefix

        print('STARTING PIPELINE RUN 1/2')
        self.dc.create_dictionary(save=save_1, load=load_1, *args, **kwargs)
        print('\n\nSTARTING PIPELINE RUN 2/2')
        dc_new.create_dictionary(save=save_2, load=load_2, *args, **kwargs)

        self.assertEqual(self.dc.progress_log, dc_new.progress_log)
        self.assertEqual(self.dc.sds_by_lang.keys(), dc_new.sds_by_lang.keys())
        for lang in self.dc.sds_by_lang.keys():
            self.assertTrue(self.dc.sds_by_lang[lang].equals(dc_new.sds_by_lang[lang]))
        self.assertDictEqual(self.dc.verses_by_bid, dc_new.verses_by_bid)

        # might fail if the aligner behaves non-deterministically
        for lang in self.dc.words_by_text_by_lang.keys():
            self.assertEqual(self.dc.words_by_text_by_lang[lang].keys(), dc_new.words_by_text_by_lang[lang].keys())
            for word in self.dc.words_by_text_by_lang[lang].values():
                self.assertEqual(word, dc_new.words_by_text_by_lang[lang][word.text])
        self.assertDictEqual(self.dc.words_by_text_by_lang, dc_new.words_by_text_by_lang)

        self.assertDictEqual(self.dc.question_by_qid_by_lang, dc_new.question_by_qid_by_lang)
        self.assertDictEqual(self.dc.wtxts_by_verse_by_bid, dc_new.wtxts_by_verse_by_bid)
        self.assertDictEqual(self.dc.aligned_wtxts_by_qid_by_lang_by_lang, dc_new.aligned_wtxts_by_qid_by_lang_by_lang)
        self.assertDictEqual(self.dc.top_scores_by_qid_by_lang, dc_new.top_scores_by_qid_by_lang)
        self.assertDictEqual(self.dc.evaluation_results_by_lang, dc_new.evaluation_results_by_lang)
        return dc_new


class TestDictionaryCreator(AbstractTestDictionaryCreator):
    def test__convert_bid_to_lang(self):
        self.assertEqual(DictionaryCreator._convert_bid_to_lang('bid-eng-DBY'), 'eng')
        self.assertEqual(DictionaryCreator._convert_bid_to_lang('bid-fra-fob'), 'fra')

    def test__group_words_by_qid(self):
        word_1 = Word('moon', 'eng', {'1.1.1.1 1'})
        word_2 = Word('lunar', 'eng', {'1.1.1.1 1'})
        word_3 = Word('star', 'eng', {'1.1.1.2 1'})
        word_4 = Word('moon star', 'eng', {'1.1.1.1 1', '1.1.1.2 1'})

        self.assertEqual(DictionaryCreator._group_words_by_qid({
            'moon': word_1,
            'lunar': word_2,
            'star': word_3,
            'moon star': word_4,
        }), {
            '1.1.1.1 1': ['moon', 'lunar', 'moon star'],
            '1.1.1.2 1': ['star', 'moon star'],
        })

    def test__transliterate_word(self):
        self.assertEqual('chandrma', DictionaryCreator._transliterate_word(self._create_word('चंद्रमा', 'hin')))
        self.assertEqual('nkshatr', DictionaryCreator._transliterate_word(self._create_word('नक्षत्र', 'hin')))
        self.assertEqual('grh', DictionaryCreator._transliterate_word(self._create_word('ग्रह', 'hin')))
        self.assertEqual('surya', DictionaryCreator._transliterate_word(self._create_word('सूर्य', 'hin')))

    def test_create_abstract_dictionary_creator(self):
        with self.assertRaises(TypeError):
            self.dc = self._create_dictionary_creator()
