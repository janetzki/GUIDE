import time
from unittest import TestCase

import networkx as nx

from dictionary_creator import DictionaryCreator


class TestDictionaryCreator(TestCase):
    def setUp(self) -> None:
        DictionaryCreator.BIBLES_BY_BID.update({
            'bid-eng-DBY-1000': '../../../dictionary_creator/data/1_test_data/eng-engDBY-1000-verses.txt',
            'bid-eng-DBY-100': '../../../dictionary_creator/test/data/eng-engDBY-100-verses.txt',
            'bid-eng-DBY-10': '../../../dictionary_creator/test/data/eng-engDBY-10-verses.txt',
            'bid-fra-fob-1000': '../../../dictionary_creator/data/1_test_data/fra-fra_fob-1000-verses.txt',
            'bid-fra-fob-100': '../../../dictionary_creator/test/data/fra-fra_fob-100-verses.txt',
            'bid-fra-fob-10': '../../../dictionary_creator/test/data/fra-fra_fob-10-verses.txt',
        })

        self.dc = DictionaryCreator(['bid-eng-DBY-10', 'bid-fra-fob-10'], score_threshold=0.2,
                                    state_files_path='test/data/0_state')
        self.maxDiff = 100000

    def _check_if_edge_weights_doubled(self):
        # check if there are only even edge weights because the edge weights doubled
        edge_weights = nx.get_edge_attributes(self.dc.word_graph, 'weight')
        for weight in edge_weights.values():
            if weight % 2 != 0:
                return False
        return True

    def _run_full_pipeline(self, dc, load, save, plot_word_lang='eng', plot_word='drink'):
        dc.create_dictionary(save=save, load=load, plot_word_lang=plot_word_lang, plot_word=plot_word)
        self.assertFalse(
            self._check_if_edge_weights_doubled())  # If this happens, there is a bug that needs to be fixed. It might be related to loading incomplete data.

    def _run_full_pipeline_twice(self, load_1, save_1, load_2, save_2, plot_word_lang='fra', plot_word='et',
                                 sd_path_prefix=None, check_isomorphism=False):
        time.sleep(1)
        dc_new = DictionaryCreator(['bid-eng-DBY-10', 'bid-fra-fob-10'], score_threshold=0.2,
                                   state_files_path='test/data/0_state')

        if sd_path_prefix is not None:
            self.dc.sd_path_prefix = sd_path_prefix
            dc_new.sd_path_prefix = sd_path_prefix

        print('STARTING PIPELINE RUN 1/2')
        self._run_full_pipeline(self.dc, load=load_1, save=save_1, plot_word_lang=plot_word_lang, plot_word=plot_word)
        print('\n\nSTARTING PIPELINE RUN 2/2')
        self._run_full_pipeline(dc_new, load=load_2, save=save_2, plot_word_lang=plot_word_lang, plot_word=plot_word)

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
        self.assertEqual(sorted(self.dc.strength_by_lang_by_wtxt_by_lang.items(), key=lambda x: str(x)),
                         sorted(dc_new.strength_by_lang_by_wtxt_by_lang.items(), key=lambda x: str(x)))
        self.assertDictEqual(self.dc.top_scores_by_qid_by_lang, dc_new.top_scores_by_qid_by_lang)
        self.assertDictEqual(self.dc.evaluation_results, dc_new.evaluation_results)

    def test_full_pipeline_without_loading_and_with_all_sds(self):
        self._run_full_pipeline_twice(load_1=False, save_1=False, load_2=False, save_2=False)

    def test_full_pipeline_with_loading_and_with_few_sds(self):
        self._run_full_pipeline_twice(load_1=False, save_1=True, load_2=True, save_2=False,
                                      plot_word_lang='fra', plot_word='et',
                                      sd_path_prefix='test/data/semdom_qa_clean_short', check_isomorphism=True)

    def test_load_only_most_current_file(self):
        self.dc.evaluation_results = {
            'eng': {'precision': 0.5}
        }
        self.dc.changed_variables.add('evaluation_results')
        self.dc._save_state()
        del self.dc.evaluation_results['eng']

        time.sleep(1)
        self.dc = DictionaryCreator(['bid-eng-DBY-10', 'bid-fra-fob-10'], score_threshold=0.2,
                                    state_files_path='test/data/0_state')
        self.dc.evaluation_results = {
            'fra': {'precision': 0.3}
        }
        self.dc.changed_variables.add('evaluation_results')
        self.dc._save_state()
        del self.dc.evaluation_results['fra']

        time.sleep(1)
        self.dc = DictionaryCreator(['bid-eng-DBY-10', 'bid-fra-fob-10'], score_threshold=0.2,
                                    state_files_path='test/data/0_state')
        self.dc._load_state()
        self.assertEqual({'fra': {'precision': 0.3}}, self.dc.evaluation_results)

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

    def test__save_state(self):
        # should not fail
        self.dc._save_state()

    def test__load_state(self):
        # should not fail
        self.dc._load_state()

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
        self._initialize_5_german_words()

        self.dc.build_word_graph()

        self.assertEqual(self.dc.word_graph.number_of_nodes(), 10)
        self.assertEqual(self.dc.word_graph.number_of_edges(), 9)

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

    def _create_word(self, text, lang, qids=None, occurrences_in_bible=1):
        word = DictionaryCreator.Word(text, lang, qids, occurrences_in_bible)
        self.dc.words_by_text_by_lang[lang][text] = word
        return word

    def _initialize_5_german_words(self):
        self.dc.target_langs = ['eng', 'deu']

        eng_word_1_1 = self._create_word('pass', 'eng',
                                         {'4.2.6.2.1 Football, soccer 9', '4.7.2 Pass laws 1', '3.6.7 Test 2'}, 30)
        eng_word_1_2 = self._create_word('passed', 'eng', None, 20)
        eng_word_1_3 = self._create_word('passing', 'eng', {'3.4 Emotion 4', '3.6.7 Test 2'}, 10)
        eng_word_2_1 = self._create_word('human being', 'eng', {'2 Person 1'}, 50)
        eng_word_2_2 = self._create_word('human beings', 'eng', {'2 Person 1'}, 40)

        deu_word_1_1 = self._create_word('vorbeigehen', 'deu', None, 30)
        deu_word_1_2 = self._create_word('vorbeigegangen', 'deu', None, 20)
        deu_word_1_3 = self._create_word('vorbeigehend', 'deu', None, 10)
        deu_word_2_1 = self._create_word('mensch', 'deu', None, 50)
        deu_word_2_2 = self._create_word('menschen', 'deu', None, 40)

        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_1, 10)
        self.dc._add_bidirectional_edge(eng_word_1_2, deu_word_1_2, 10)
        self.dc._add_bidirectional_edge(eng_word_1_3, deu_word_1_3, 10)
        self.dc._add_bidirectional_edge(eng_word_2_1, deu_word_2_1, 10)
        self.dc._add_bidirectional_edge(eng_word_2_2, deu_word_2_2, 10)

        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_2, 5)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_3, 5)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_2_2, 1)
        self.dc._add_bidirectional_edge(eng_word_2_1, deu_word_2_2, 5)

    def test__predict_lemmas(self):
        self._initialize_5_german_words()

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
        # self._create_word('drink', 'eng', None, 3)
        # self._create_word('drank', 'eng', None, 2)
        # self._create_word('drunk', 'eng', None, 1)
        # self._create_word('water', 'eng', None, 5)
        # self._create_word('waters', 'eng', None, 4)

        self._create_word('boire', 'fra', None, 3)
        self._create_word('bu', 'fra', None, 2)
        self._create_word('buve', 'fra', None, 1)
        self._create_word('eau', 'fra', None, 5)
        self._create_word('eaux', 'fra', None, 4)

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

        self.dc._contract_lemmas()
        self.dc.build_word_graph()
        self.dc.plot_subgraph('fra', 'boire')

        self.assertEqual(self.dc.words_by_text_by_lang['fra'].keys(), {'boire', 'eau'})
        self.assertEqual(self.dc.words_by_text_by_lang['fra']['boire'].display_text, 'BOIRE (3)')
        self.assertEqual(self.dc.words_by_text_by_lang['fra']['eau'].display_text, 'EAU (2)')

    def test_predict_links(self):
        self._initialize_5_german_words()

        self.dc.build_word_graph()
        self.dc.predict_links()
        self.dc.plot_subgraph('eng', 'pass')

        self.assertDictEqual({
            'deu': {
                '2 Person 1': {
                    'mensch': 0.8,
                    'menschen': 0.7692307692307693
                },
                '3.4 Emotion 4': {
                    'vorbeigehend': 0.8
                },
                '3.6.7 Test 2': {
                    'menschen': 0.05405405405405406,
                    'vorbeigegangen': 0.2777777777777778,
                    'vorbeigehen': 0.6451612903225806,
                    'vorbeigehend': 0.8
                },
                '4.2.6.2.1 Football, soccer 9': {
                    'menschen': 0.05405405405405406,
                    'vorbeigegangen': 0.2777777777777778,
                    'vorbeigehen': 0.6451612903225806,
                    'vorbeigehend': 0.2777777777777778
                },
                '4.7.2 Pass laws 1': {
                    'menschen': 0.05405405405405406,
                    'vorbeigegangen': 0.2777777777777778,
                    'vorbeigehen': 0.6451612903225806,
                    'vorbeigehend': 0.2777777777777778
                }}
        }, dict(self.dc.top_scores_by_qid_by_lang))

        # link_candidates = [
        #     (self.words_by_text_by_lang['fra']['eau'],
        #      self.words_by_text_by_lang['deu']['wasser']),
        #     (self.words_by_text_by_lang['eng']['water'],
        #      self.words_by_text_by_lang['deu']['wasser']),
        #     (self.words_by_text_by_lang['eng']['water'],
        #      self.words_by_text_by_lang['fra']['eau']),
        #
        #     (self.words_by_text_by_lang['fra']['boire'],
        #      self.words_by_text_by_lang['deu']['trinken']),
        #     (self.words_by_text_by_lang['eng']['drink'],
        #      self.words_by_text_by_lang['deu']['trinken']),
        #     (self.words_by_text_by_lang['eng']['drink'],
        #      self.words_by_text_by_lang['fra']['boire']),
        # ]

    def test_train_tfidf_based_model(self):
        self.dc.aligned_wtxts_by_qid_by_lang_by_lang = {
            'eng': {
                'eng': {
                    '4.9.6 1': ', heaven, and, heaven',
                    '1.2 1': ', earth, earth, earth',
                }
            },
            'fra': {
                'eng': {
                    '1.2.3 1': ', eaux, eaux, les, eaux, les, eaux, les, eaux, avec, les, eaux, les, eaux, eaux',
                    '8.3.3.2 6': ', ténèbres, ténèbres, ténèbres',
                }
            }
        }

        self.dc.train_tfidf_based_model()

        self.assertEqual({
            'eng': {'1.2 1': {'earth': 1.0},
                    '4.9.6 1': {'and': 0.447213595499958,
                                'heaven': 0.894427190999916}},
            'fra': {'1.2.3 1': {'avec': 0.10540925533894598,
                                'eaux': 0.8432740427115678,
                                'les': 0.5270462766947299},
                    '8.3.3.2 6': {'ténèbres': 1.0}}
        }, self.dc.top_scores_by_qid_by_lang)

    # def test__filter_target_sds_with_threshold(self):
    #     self.fail()

    # def test__compute_f1_score(self):
    #     self.fail()

    # def test__load_test_data(self):
    #     self.fail()

    def test__compute_mean_reciprocal_rank(self):
        self._create_word('moon', 'eng', {'1.1.1.1 1'})
        self._create_word('lunar', 'eng', {'1.1.1.1 1'})
        self._create_word('star', 'eng', {'1.1.1.2 1'})
        self._create_word('moon star', 'eng', {'1.1.1.1 1', '1.1.1.2 1'})

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
