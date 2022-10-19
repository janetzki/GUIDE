from unittest import TestCase

import networkx as nx

from dictionary_creator import DictionaryCreator


class TestDictionaryCreator(TestCase):
    def setUp(self) -> None:
        self.dc = DictionaryCreator(bibles_by_bid={
            'bid-eng-DBY-10': '../../../dictionary_creator/test/data/eng-engDBY-10-verses.txt',
            'bid-fra-fob-10': '../../../dictionary_creator/test/data/fra-fra_fob-10-verses.txt',
        },
            score_threshold=0.2,
        )
        self.maxDiff = 100000

    def _check_if_edge_weights_doubled(self):
        # check if there are only even edge weights because the edge weights doubled
        edge_weights = nx.get_edge_attributes(self.dc.word_graph, 'weight')
        for weight in edge_weights.values():
            if weight % 2 != 0:
                return False
        return True

    def _run_full_pipeline(self, dc, load, save):
        dc.preprocess_data(load=load, save=save)
        dc.map_words_to_qids(load=load, save=save)

        dc.build_word_graph(load=load, save=save)  # build the graph with single words as nodes
        dc._predict_lemmas(load=load, save=save)
        dc._contract_lemmas(load=load, save=save)
        dc.build_word_graph(load=load, save=save)  # build the word graph with lemma groups as nodes
        dc.predict_links(load=load, save=save)
        dc.plot_subgraph(lang='fra', text='et', min_count=3)
        self.assertFalse(
            self._check_if_edge_weights_doubled())  # If this happens, there is a bug that needs to be fixed. It might be related to loading incomplete data.

        # dc.train_tfidf_based_model(load=load, save=save)
        return dc.evaluate(load=load, print_reciprocal_ranks=False)

    def _run_full_pipeline_twice(self, load_1, save_1, load_2, save_2, sd_path_prefix=None, check_isomorphism=False):
        dc_new = DictionaryCreator(bibles_by_bid={
            'bid-eng-DBY-10': '../../../dictionary_creator/test/data/eng-engDBY-10-verses.txt',
            'bid-fra-fob-10': '../../../dictionary_creator/test/data/fra-fra_fob-10-verses.txt',
        },
            score_threshold=0.2,
        )

        if sd_path_prefix is not None:
            self.dc.sd_path_prefix = sd_path_prefix
            dc_new.sd_path_prefix = sd_path_prefix

        print('STARTING PIPELINE RUN 1/2')
        evaluation_results_run_1 = self._run_full_pipeline(self.dc, load=load_1, save=save_1)
        print('\n\nSTARTING PIPELINE RUN 2/2')
        evaluation_results_run_2 = self._run_full_pipeline(dc_new, load=load_2, save=save_2)

        self.assertEqual(self.dc.sds_by_lang.keys(), dc_new.sds_by_lang.keys())
        for lang in self.dc.sds_by_lang.keys():
            self.assertTrue(self.dc.sds_by_lang[lang].equals(dc_new.sds_by_lang[lang]))
        self.assertDictEqual(self.dc.verses_by_bid, dc_new.verses_by_bid)
        self.assertDictEqual(self.dc.words_by_text_by_lang['eng'], dc_new.words_by_text_by_lang['eng'])
        self.assertDictEqual(self.dc.question_by_qid_by_lang, dc_new.question_by_qid_by_lang)
        self.assertDictEqual(self.dc.wtxts_by_verse_by_bid, dc_new.wtxts_by_verse_by_bid)
        self.assertDictEqual(self.dc.aligned_wtxts_by_qid_by_lang_by_lang, dc_new.aligned_wtxts_by_qid_by_lang_by_lang)
        if check_isomorphism:
            self.assertTrue(nx.is_isomorphic(self.dc.word_graph, dc_new.word_graph,
                                             edge_match=lambda x, y: x['weight'] == y['weight']))
        self.assertDictEqual(self.dc.base_lemma_by_wtxt_by_lang, dc_new.base_lemma_by_wtxt_by_lang)
        self.assertDictEqual(self.dc.lemma_group_by_base_lemma_by_lang, dc_new.lemma_group_by_base_lemma_by_lang)
        self.assertEqual(sorted(self.dc.strength_by_lang_by_word.items(), key=lambda x: str(x)),
                         sorted(dc_new.strength_by_lang_by_word.items(), key=lambda x: str(x)))
        self.assertDictEqual(self.dc.top_scores_by_qid_by_lang, dc_new.top_scores_by_qid_by_lang)
        self.assertDictEqual(evaluation_results_run_1, evaluation_results_run_2)

    def test_full_pipeline_with_loading(self):
        self._run_full_pipeline_twice(load_1=False, save_1=False, load_2=False, save_2=False)

    def test_full_pipeline_without_loading(self):
        self._run_full_pipeline_twice(load_1=False, save_1=True, load_2=True, save_2=False,
                                      sd_path_prefix='test/data/semdom_qa_clean_short', check_isomorphism=True)

    def test__convert_bid_to_lang(self):
        self.assertEqual(DictionaryCreator._convert_bid_to_lang('bid-eng-DBY'), 'eng')
        self.assertEqual(DictionaryCreator._convert_bid_to_lang('bid-fra-fob'), 'fra')

    def test__group_words_by_qid(self):
        word_1 = DictionaryCreator.Word('moon', 'eng', {'1.1.1.1 1'})
        word_2 = DictionaryCreator.Word('lunar', 'eng', {'1.1.1.1 1'})
        word_3 = DictionaryCreator.Word('star', 'eng', {'1.1.1.2 1'})
        word_4 = DictionaryCreator.Word('moon star', 'eng', {'1.1.1.1 1', '1.1.1.2 1'})

        self.assertEqual(DictionaryCreator._group_words_by_qid({
            'moon': word_1,
            'lunar': word_2,
            'star': word_3,
            'moon star': word_4,
        }), {
            '1.1.1.1 1': ['moon', 'lunar', 'moon star'],
            '1.1.1.2 1': ['star', 'moon star'],
        })

    # def test__transliterate_word(self):
    #     self.fail()

    # def test__apply_prediction(self):
    #     self.fail()

    # def test__weighted_resource_allocation_index(self):
    #     self.fail()

    # def test__save_state(self):
    #     self.fail()

    # def test__load_state(self):
    #     self.fail()

    # def test__load_data(self):
    #     self.fail()

    # def test__build_sds(self):
    #     self.fail()

    def test__lemmatize_verse(self):
        result = self.dc._lemmatize_english_verse(['I', 'drank', 'water', '.'])
        self.assertEqual(result, ['I', 'drink', 'water', '.'])

    # def test__tokenize_verses(self):
    #     self.fail()

    # def test__combine_alignments(self):
    #     self.fail()

    # def test_preprocess_data(self):
    #     self.fail()

    # def test__add_directed_edge(self):
    #     self.fail()

    # def test__map_word_to_qid(self):
    #     self.fail()

    # def test__map_word_to_qid_bidirectionally(self):
    #     self.fail()

    # def test__map_two_bibles(self):
    #     self.fail()

    # def test_map_words_to_qids(self):
    #     self.fail()

    def test_build_word_graph(self):
        # todo: fix and test 'gather' bug
        self.fail()

    # def test_plot_subgraph(self):
    #     self.fail()

    # def test_check_if_edge_weights_doubled(self):
    #     self.fail()

    # def test__find_lemma_link_candidates(self):
    #     self.fail()

    # def test__find_translation_link_candidates(self):
    #     self.fail()

    # def test__compute_sum_of_weights(self):
    #     self.fail()

    # def test__compute_link_score(self):
    #     self.fail()

    def test__predict_lemmas(self):
        self.dc.target_langs = ['eng', 'deu']

        eng_word_1_1 = self.dc.words_by_text_by_lang['eng']['pass'] = DictionaryCreator.Word('pass', 'eng', set(), 3)
        eng_word_1_2 = self.dc.words_by_text_by_lang['eng']['passed'] = DictionaryCreator.Word('passed', 'eng', set(),
                                                                                               2)
        eng_word_1_3 = self.dc.words_by_text_by_lang['eng']['passing'] = DictionaryCreator.Word('passing', 'eng', set(),
                                                                                                1)
        eng_word_2_1 = self.dc.words_by_text_by_lang['eng']['human'] = DictionaryCreator.Word('human', 'eng', set(), 5)
        eng_word_2_2 = self.dc.words_by_text_by_lang['eng']['humans'] = DictionaryCreator.Word('humans', 'eng', set(),
                                                                                               4)

        deu_word_1_1 = self.dc.words_by_text_by_lang['deu']['vorbeigehen'] = DictionaryCreator.Word('vorbeigehen',
                                                                                                    'deu', set(), 3)
        deu_word_1_2 = self.dc.words_by_text_by_lang['deu']['vorbeigegangen'] = DictionaryCreator.Word('vorbeigegangen',
                                                                                                       'deu', set(), 2)
        deu_word_1_3 = self.dc.words_by_text_by_lang['deu']['vorbeigehend'] = DictionaryCreator.Word('vorbeigehend',
                                                                                                     'deu', set(), 1)
        deu_word_2_1 = self.dc.words_by_text_by_lang['deu']['mensch'] = DictionaryCreator.Word('mensch', 'deu', set(),
                                                                                               5)
        deu_word_2_2 = self.dc.words_by_text_by_lang['deu']['menschen'] = DictionaryCreator.Word('menschen', 'deu',
                                                                                                 set(), 4)

        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_1)
        self.dc._add_bidirectional_edge(eng_word_1_2, deu_word_1_2)
        self.dc._add_bidirectional_edge(eng_word_1_3, deu_word_1_3)
        self.dc._add_bidirectional_edge(eng_word_2_1, deu_word_2_1)
        self.dc._add_bidirectional_edge(eng_word_2_2, deu_word_2_2)

        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_1)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_2)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_3)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_2_2)
        self.dc._add_bidirectional_edge(eng_word_2_1, deu_word_2_2)

        self.dc.build_word_graph()
        self.dc.plot_subgraph('eng', 'pass')
        self.dc._predict_lemmas()

        self.assertEqual({
            'vorbeigehen': 'vorbeigehen',
            'vorbeigegangen': 'vorbeigehen',
            'vorbeigehend': 'vorbeigehen',
            'mensch': 'mensch',
            'menschen': 'mensch',
        }, self.dc.base_lemma_by_wtxt_by_lang['deu'])
        self.assertEqual(
            {'vorbeigehen': {'vorbeigehen', 'vorbeigegangen', 'vorbeigehend'}, 'mensch': {'mensch', 'menschen'}},
            self.dc.lemma_group_by_base_lemma_by_lang['deu'])

    def test__contract_lemmas(self):
        self.dc.words_by_text_by_lang['fra']['boire'] = DictionaryCreator.Word('boire', 'fra', set(), 3)
        self.dc.words_by_text_by_lang['fra']['bu'] = DictionaryCreator.Word('bu', 'fra', set(), 2)
        self.dc.words_by_text_by_lang['fra']['buve'] = DictionaryCreator.Word('buve', 'fra', set(), 1)
        self.dc.words_by_text_by_lang['fra']['eau'] = DictionaryCreator.Word('eau', 'fra', set(), 5)
        self.dc.words_by_text_by_lang['fra']['eaux'] = DictionaryCreator.Word('eaux', 'fra', set(), 4)

        self.dc.base_lemma_by_wtxt_by_lang = {
            'eng': {
                # 'drink': 'drink',
                # 'drank': 'drink',
                # 'drunk': 'drink',
                # 'water': 'water',
            },
            'fra': {
                'boire': 'boire',
                'bu': 'boire',
                'buve': 'boire',
                'eau': 'eau',
                'eaux': 'eau',
            }
        }
        self.dc.lemma_group_by_base_lemma_by_lang = {
            'eng': {
                # 'drink': ['drink', 'drank', 'drunk'],
                # 'water': ['water'],
            },
            'fra': {
                'boire': {'boire', 'bu', 'buve'},
                'eau': {'eau', 'eaux'},
            }
        }

        self.dc._contract_lemmas(load=False, save=False)

        self.assertEqual(self.dc.words_by_text_by_lang['fra'].keys(), {'boire', 'eau'})
        self.assertEqual(self.dc.words_by_text_by_lang['fra']['boire'].display_text, 'BOIRE (3)')
        self.assertEqual(self.dc.words_by_text_by_lang['fra']['eau'].display_text, 'EAU (2)')

    # def test_predict_links(self):
    #     self.fail()

    # def test_train_tfidf_based_model(self):
    #     self.fail()

    # def test__filter_target_sds_with_threshold(self):
    #     self.fail()

    # def test__compute_f1_score(self):
    #     self.fail()

    # def test__load_test_data(self):
    #     self.fail()

    def test__compute_mean_reciprocal_rank(self):
        word_1 = DictionaryCreator.Word('moon', 'eng', {'1.1.1.1 1'})
        word_2 = DictionaryCreator.Word('lunar', 'eng', {'1.1.1.1 1'})
        word_3 = DictionaryCreator.Word('star', 'eng', {'1.1.1.2 1'})
        word_4 = DictionaryCreator.Word('moon star', 'eng', {'1.1.1.1 1', '1.1.1.2 1'})

        self.dc.words_by_text_by_lang['eng']['moon'] = word_1
        self.dc.words_by_text_by_lang['eng']['lunar'] = word_2
        self.dc.words_by_text_by_lang['eng']['star'] = word_3
        self.dc.words_by_text_by_lang['eng']['moon star'] = word_4

        self.dc.question_by_qid_by_lang['eng']['1.1.1.1 1'] = 'What words refer to the moon?'
        self.dc.question_by_qid_by_lang['eng']['1.1.1.2 1'] = 'What words are used to refer to the stars?'

        self.dc.top_scores_by_qid_by_lang = {
            'eng': {
                '1.1.1.1 1': {'moon': 0.5, 'lunar': 0.4, 'moon star': 0.3},
                '1.1.1.2 1': {'moon star': 0.2, 'star': 0.1},
            }
        }

        mean_reciprocal_rank = self.dc._compute_mean_reciprocal_rank('eng', True)

        self.assertEqual(0.75, mean_reciprocal_rank)

    # def test_evaluate(self):
    #     self.fail()
