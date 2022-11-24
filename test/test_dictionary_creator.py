import os
from unittest import TestCase

import networkx as nx

from dictionary_creator import DictionaryCreator
from word import Word


class TestDictionaryCreator(TestCase):
    def setUp(self) -> None:
        self.tested_class = DictionaryCreator  # overwritten by child classes
        self.dc = None  # overwritten by child classes

        # delete all files in test/data/0_state
        for file in os.listdir('../test/data/0_state'):
            os.remove('../test/data/0_state/' + file)

        # delete all files in test/data/1_aligned_bibles
        for file in os.listdir('../test/data/1_aligned_bibles'):
            os.remove('../test/data/1_aligned_bibles/' + file)

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

    def _create_dictionary_creator(self, bids=None, sd_path_prefix='test/data/semdom_qa_clean_short'):
        if bids is None:
            bids = ['bid-eng-DBY-10', 'bid-fra-fob-10']
        dc = self.tested_class(bids, score_threshold=0.2,
                               base_path='..',
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

    def _check_if_edge_weights_doubled(self):
        # check if there are only even edge weights because the edge weights doubled
        edge_weights = nx.get_edge_attributes(self.dc.word_graph, 'weight')
        for weight in edge_weights.values():
            if weight % 2 != 0:
                return False
        return True

    def _run_full_pipeline(self, dc, load, save, plot_word_lang='eng', plot_word='drink', min_count=1):
        dc.create_dictionary(save=save, load=load, plot_word_lang=plot_word_lang, plot_wtxt=plot_word,
                             min_count=min_count)
        self.assertFalse(self._check_if_edge_weights_doubled())
        # If this happens, there is a bug that needs to be fixed. It might be related to loading incomplete data.

    def _run_full_pipeline_twice(self, load_1, save_1, load_2, save_2, plot_word_lang='fra', plot_word='et',
                                 min_count=1, sd_path_prefix=None, check_isomorphism=False):
        dc_new = self._create_dictionary_creator()

        if sd_path_prefix is not None:
            self.dc.sd_path_prefix = sd_path_prefix
            dc_new.sd_path_prefix = sd_path_prefix

        print('STARTING PIPELINE RUN 1/2')
        self._run_full_pipeline(self.dc, load=load_1, save=save_1, plot_word_lang=plot_word_lang, plot_word=plot_word,
                                min_count=min_count)
        print('\n\nSTARTING PIPELINE RUN 2/2')
        self._run_full_pipeline(dc_new, load=load_2, save=save_2, plot_word_lang=plot_word_lang, plot_word=plot_word,
                                min_count=min_count)

        self.assertEqual(self.dc.progress_log, dc_new.progress_log)
        self.assertEqual(self.dc.sds_by_lang.keys(), dc_new.sds_by_lang.keys())
        for lang in self.dc.sds_by_lang.keys():
            self.assertTrue(self.dc.sds_by_lang[lang].equals(dc_new.sds_by_lang[lang]))
        self.assertDictEqual(self.dc.verses_by_bid, dc_new.verses_by_bid)
        self.assertDictEqual(self.dc.words_by_text_by_lang['fra']["'"]._aligned_words,
                             dc_new.words_by_text_by_lang['fra']["'"]._aligned_words)
        self.assertEqual(self.dc.words_by_text_by_lang['eng'].keys(), dc_new.words_by_text_by_lang['eng'].keys())
        self.assertEqual(next(iter(self.dc.words_by_text_by_lang['eng'].keys())),
                         next(iter(dc_new.words_by_text_by_lang['eng'].keys())))
        for word in self.dc.words_by_text_by_lang['eng'].values():
            self.assertEqual(word, dc_new.words_by_text_by_lang['eng'][word.text])
        self.assertDictEqual(self.dc.words_by_text_by_lang, dc_new.words_by_text_by_lang)
        self.assertDictEqual(self.dc.question_by_qid_by_lang, dc_new.question_by_qid_by_lang)
        self.assertDictEqual(self.dc.wtxts_by_verse_by_bid, dc_new.wtxts_by_verse_by_bid)
        self.assertDictEqual(self.dc.aligned_wtxts_by_qid_by_lang_by_lang, dc_new.aligned_wtxts_by_qid_by_lang_by_lang)
        if check_isomorphism:
            self.assertTrue(nx.is_isomorphic(self.dc.word_graph, dc_new.word_graph,
                                             edge_match=lambda x, y: x['weight'] == y['weight']))
        self.assertDictEqual(self.dc.base_lemma_by_wtxt_by_lang, dc_new.base_lemma_by_wtxt_by_lang)
        self.assertDictEqual(self.dc.lemma_group_by_base_lemma_by_lang, dc_new.lemma_group_by_base_lemma_by_lang)
        self.assertEqual(sorted(self.dc.strength_by_lang_by_wtxt_by_lang.items(), key=lambda x: str(x)),
                         sorted(dc_new.strength_by_lang_by_wtxt_by_lang.items(), key=lambda x: str(x)))
        self.assertDictEqual(self.dc.top_scores_by_qid_by_lang, dc_new.top_scores_by_qid_by_lang)
        self.assertDictEqual(self.dc.evaluation_results_by_lang, dc_new.evaluation_results_by_lang)


class TestDictionaryCreatorFast(TestDictionaryCreator):
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
