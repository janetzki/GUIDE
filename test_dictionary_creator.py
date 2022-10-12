from unittest import TestCase

import networkx as nx

from dictionary_creator import DictionaryCreator


class TestDictionaryCreator(TestCase):
    def setUp(self) -> None:
        self.dc = DictionaryCreator(bibles_by_bid={
            'bid-eng-DBY-1000': '../../../dictionary_creator/data/1_test_data/eng-engDBY-1000-verses.txt',
            'bid-fra-fob-1000': '../../../dictionary_creator/data/1_test_data/fra-fra_fob-1000-verses.txt',
        }, score_threshold=0.2)

    def check_if_edge_weights_doubled(self):
        # check if there are only even edge weights because the edge weights doubled
        edge_weights = nx.get_edge_attributes(self.dc.word_graph, 'weight')
        for weight in edge_weights.values():
            if weight % 2 != 0:
                return False
        return True

    def execute_full_pipeline(self, load, save):
        self.dc.preprocess_data(load=load, save=save)
        self.dc.map_words_to_qids(load=load, save=save)

        self.dc.build_word_graph(load=load, save=save)  # build the graph with single words as nodes
        self.dc._predict_lemmas(load=load, save=save)
        self.dc._save_state()
        self.dc._contract_lemmas(load=load, save=save)
        self.dc.build_word_graph(load=load, save=save)  # build the word graph with lemma groups as nodes
        self.dc.predict_links(load=load, save=save)
        self.dc.plot_subgraph(lang='eng', text='drink', min_count=1)
        self.assertFalse(
            self.check_if_edge_weights_doubled())  # If this happens, there is a bug that needs to be fixed. It might be related to loading incomplete data.

        # dc.train_tfidf_based_model(load=load, save=save)
        self.dc.evaluate(load=load, print_reciprocal_ranks=False)
        self.dc._save_state()

    def test_full_pipeline(self):
        self.execute_full_pipeline(load=False, save=False)
        self.execute_full_pipeline(load=True, save=False)

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
    #
    # def test__apply_prediction(self):
    #     self.fail()
    #
    # def test__weighted_resource_allocation_index(self):
    #     self.fail()
    #
    # def test__save_state(self):
    #     self.fail()
    #
    # def test__load_state(self):
    #     self.fail()
    #
    # def test__load_data(self):
    #     self.fail()
    #
    # def test__build_sds(self):
    #     self.fail()
    #
    # def test__lemmatize_verse(self):
    #     self.fail()
    #
    # def test__tokenize_verses(self):
    #     self.fail()
    #
    # def test__combine_alignments(self):
    #     self.fail()
    #
    # def test_preprocess_data(self):
    #     self.fail()
    #
    # def test__add_directed_edge(self):
    #     self.fail()
    #
    # def test__map_word_to_qid(self):
    #     self.fail()
    #
    # def test__map_word_to_qid_bidirectionally(self):
    #     self.fail()
    #
    # def test__map_two_bibles(self):
    #     self.fail()
    #
    # def test_map_words_to_qids(self):
    #     self.fail()
    #
    # def test_build_word_graph(self):
    #     self.fail()
    #
    # def test_plot_subgraph(self):
    #     self.fail()
    #
    # def test_check_if_edge_weights_doubled(self):
    #     self.fail()
    #
    # def test__find_lemma_link_candidates(self):
    #     self.fail()
    #
    # def test__find_translation_link_candidates(self):
    #     self.fail()
    #
    # def test__compute_sum_of_weights(self):
    #     self.fail()
    #
    # def test__compute_link_score(self):
    #     self.fail()
    #
    # def test__predict_lemmas(self):
    #     self.fail()
    #
    # def test__contract_lemmas(self):
    #     self.fail()
    #
    # def test_predict_links(self):
    #     self.fail()
    #
    # def test_train_tfidf_based_model(self):
    #     self.fail()
    #
    # def test__filter_target_sds_with_threshold(self):
    #     self.fail()
    #
    # def test__compute_f1_score(self):
    #     self.fail()
    #
    # def test__load_test_data(self):
    #     self.fail()
    #
    # def test__compute_mean_reciprocal_rank(self):
    #     self.fail()
    #
    # def test_evaluate(self):
    #     self.fail()
