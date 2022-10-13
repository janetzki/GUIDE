from unittest import TestCase

import networkx as nx

from dictionary_creator import DictionaryCreator


class TestDictionaryCreator(TestCase):
    def setUp(self) -> None:
        self.dc = DictionaryCreator(bibles_by_bid={
            'bid-eng-DBY-100': '../../../dictionary_creator/data/1_test_data/eng-engDBY-100-verses.txt',
            'bid-fra-fob-100': '../../../dictionary_creator/data/1_test_data/fra-fra_fob-100-verses.txt',
            # 'bid-eng-DBY-1000': '../../../dictionary_creator/data/1_test_data/eng-engDBY-1000-verses.txt',
            # 'bid-fra-fob-1000': '../../../dictionary_creator/data/1_test_data/fra-fra_fob-1000-verses.txt',
        }, score_threshold=0.2)

    def check_if_edge_weights_doubled(self):
        # check if there are only even edge weights because the edge weights doubled
        edge_weights = nx.get_edge_attributes(self.dc.word_graph, 'weight')
        for weight in edge_weights.values():
            if weight % 2 != 0:
                return False
        return True

    def execute_full_pipeline(self, dc, load, save):
        dc.preprocess_data(load=load, save=save)
        dc.map_words_to_qids(load=load, save=save)

        dc.build_word_graph(load=load, save=save)  # build the graph with single words as nodes
        dc._predict_lemmas(load=load, save=save)
        dc._contract_lemmas(load=load, save=save)
        dc.build_word_graph(load=load, save=save)  # build the word graph with lemma groups as nodes
        dc.predict_links(load=load, save=save)
        dc.plot_subgraph(lang='eng', text='drink', min_count=1)
        dc.plot_subgraph(lang='fra', text='et', min_count=1)
        self.assertFalse(
            self.check_if_edge_weights_doubled())  # If this happens, there is a bug that needs to be fixed. It might be related to loading incomplete data.

        # dc.train_tfidf_based_model(load=load, save=save)
        evaluation_results = dc.evaluate(load=load, print_reciprocal_ranks=False)
        return evaluation_results

    def test_full_pipeline_twice(self):
        print('STARTING PIPELINE RUN 1/2')
        evaluation_results_run_1 = self.execute_full_pipeline(self.dc, load=False, save=False)

        dc_new = DictionaryCreator(bibles_by_bid={
            'bid-eng-DBY-1000': '../../../dictionary_creator/data/1_test_data/eng-engDBY-1000-verses.txt',
            'bid-fra-fob-1000': '../../../dictionary_creator/data/1_test_data/fra-fra_fob-1000-verses.txt',
        }, score_threshold=0.2)

        print('\n\nSTARTING PIPELINE RUN 2/2')
        evaluation_results_run_2 = self.execute_full_pipeline(dc_new, load=True, save=True)
        self.assertEqual(evaluation_results_run_1, evaluation_results_run_2)

    def test__convert_bid_to_lang(self):
        self.assertEqual(DictionaryCreator._convert_bid_to_lang('bid-eng-DBY'), 'eng')
        self.assertEqual(DictionaryCreator._convert_bid_to_lang('bid-fra-fob'), 'fra')

    def test__group_words_by_qid(self):
        word_1 = DictionaryCreator.Word('moon', 'eng', {'1.1.1.1 Moon'})
        word_2 = DictionaryCreator.Word('lunar', 'eng', {'1.1.1.1 Moon'})
        word_3 = DictionaryCreator.Word('star', 'eng', {'1.1.1.2 Star'})
        word_4 = DictionaryCreator.Word('moon star', 'eng', {'1.1.1.1 Moon', '1.1.1.2 Star'})

        self.assertEqual(DictionaryCreator._group_words_by_qid({
            'moon': word_1,
            'lunar': word_2,
            'star': word_3,
            'moon star': word_4,
        }), {
            '1.1.1.1 Moon': ['moon', 'lunar', 'moon star'],
            '1.1.1.2 Star': ['star', 'moon star'],
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

    # def test_build_word_graph(self):
    #     self.fail()

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
        self.dc.words_by_text_by_lang['fra']['boire'] = word_1 = DictionaryCreator.Word('boire', 'fra', set(), 3)
        self.dc.words_by_text_by_lang['fra']['bu'] = word_2 = DictionaryCreator.Word('bu', 'fra', set(), 2)
        self.dc.words_by_text_by_lang['fra']['buve'] = word_3 = DictionaryCreator.Word('buve', 'fra', set(), 1)
        self.dc.words_by_text_by_lang['fra']['eau'] = word_4 = DictionaryCreator.Word('eau', 'fra', set(), 5)
        self.dc.words_by_text_by_lang['fra']['eaux'] = word_5 = DictionaryCreator.Word('eaux', 'fra', set(), 4)
        word_1.add_aligned_word(word_2)
        word_1.add_aligned_word(word_3)
        word_4.add_aligned_word(word_5)

        self.dc.build_word_graph()
        self.dc._predict_lemmas()

        self.assertEqual({
            'boire': 'boire',
            'bu': 'bu',
            'buve': 'buve',
            'eau': 'eau',
            'eaux': 'eaux',
        }, self.dc.base_lemma_by_wtxt_by_lang['fra'], )
        self.assertEqual({'boire': {'boire'}, 'bu': {'bu'}, 'buve': {'buve'}, 'eau': {'eau'}, 'eaux': {'eaux'}},
                         self.dc.lemma_group_by_base_lemma_by_lang['fra'])

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

    # def test__compute_mean_reciprocal_rank(self):
    #     self.fail()

    # def test_evaluate(self):
    #     self.fail()
