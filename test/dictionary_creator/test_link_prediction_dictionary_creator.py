import os

import networkx as nx

from src.dictionary_creator.link_prediction_dictionary_creator import LinkPredictionDictionaryCreator
from test.dictionary_creator.test_dictionary_creator import AbstractTestDictionaryCreator


class TestLinkPredictionDictionaryCreator(AbstractTestDictionaryCreator):
    @staticmethod
    def _check_if_edge_weights_doubled(dc):
        # check if there are only even edge weights because the edge weights doubled
        edge_weights = nx.get_edge_attributes(dc.word_graph, 'weight')
        for weight in edge_weights.values():
            if weight % 2 != 0:
                return False
        return True

    def _run_full_pipeline_twice(self, check_isomorphism=True, *args, **kwargs):
        dc_new = super()._run_full_pipeline_twice(*args, **kwargs)
        if check_isomorphism:
            self.assertTrue(nx.is_isomorphic(self.dc.word_graph, dc_new.word_graph,
                                             edge_match=lambda x, y: x['weight'] == y['weight']))
        self.assertDictEqual(self.dc.base_lemma_by_wtxt_by_lang, dc_new.base_lemma_by_wtxt_by_lang)
        self.assertDictEqual(self.dc.lemma_group_by_base_lemma_by_lang, dc_new.lemma_group_by_base_lemma_by_lang)
        self.assertEqual(sorted(self.dc.strength_by_lang_by_wtxt_by_lang.items(), key=lambda x: str(x)),
                         sorted(dc_new.strength_by_lang_by_wtxt_by_lang.items(), key=lambda x: str(x)))
        self._check_if_edge_weights_doubled(self.dc)
        self._check_if_edge_weights_doubled(dc_new)
        return dc_new

    def setUp(self) -> None:
        super().setUp()
        self.tested_class = LinkPredictionDictionaryCreator
        self.dc = self._create_dictionary_creator()


class TestLinkPredictionDictionaryCreatorFast(TestLinkPredictionDictionaryCreator):
    def test_load_only_most_current_file(self):
        self.dc.evaluation_results_by_lang = {
            'eng': {'precision': 0.5}
        }
        self.dc._save_state()
        del self.dc.evaluation_results_by_lang['eng']

        self.dc = self._create_dictionary_creator()

        self.dc.evaluation_results_by_lang = {
            'fra': {'precision': 0.3}
        }
        self.dc._save_state()
        del self.dc.evaluation_results_by_lang['fra']

        self.dc = self._create_dictionary_creator()

        self.dc._load_state()
        self.assertEqual({'fra': {'precision': 0.3}}, self.dc.evaluation_results_by_lang)

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

    def test__load_state_with_broken_file(self):
        # create directory
        directory = os.path.join(self.dc.state_files_base_path, str(int(self.dc.start_timestamp) - 1))
        os.makedirs(directory)

        # create a broken dill file
        file_path = os.path.join(directory, 'word_graph.dill')
        with open(file_path, 'w') as f:
            f.write('broken file')

        self.dc._load_state()

        self.assertEqual(None, self.dc.word_graph)

    def test__load_data(self):
        self.dc._load_data()

        self.assertEqual({'eng', 'fra'}, self.dc.sds_by_lang.keys())
        self.assertEqual(9, len(self.dc.sds_by_lang['eng']))  # 7955 with all sds
        self.assertEqual(9, len(self.dc.sds_by_lang['fra']))  # 7812 with all sds

        self.assertEqual({'bid-eng-DBY-10', 'bid-fra-fob-10'}, self.dc.verses_by_bid.keys())
        self.assertEqual('In the beginning God created the heavens and the earth.\n',
                         self.dc.verses_by_bid['bid-eng-DBY-10'][0])
        self.assertEqual('Au commencement, Dieu créa les cieux et la terre.\n',
                         self.dc.verses_by_bid['bid-fra-fob-10'][0])

    def test__load_data_with_missing_sds(self):
        self.dc = self._create_dictionary_creator(['bid-eng-DBY-10', 'bid-deu-10'])

        self.dc._load_data()

        self.assertEqual({'eng', 'deu'}, self.dc.sds_by_lang.keys())
        self.assertEqual(9, len(self.dc.sds_by_lang['eng']))
        self.assertEqual(0, len(self.dc.sds_by_lang['deu']))

        self.assertEqual({'bid-eng-DBY-10', 'bid-deu-10'}, self.dc.verses_by_bid.keys())
        self.assertEqual('In the beginning God created the heavens and the earth.\n',
                         self.dc.verses_by_bid['bid-eng-DBY-10'][0])
        self.assertEqual('Im Anfang schuf Gott die Himmel und die Erde.\n', self.dc.verses_by_bid['bid-deu-10'][0])

    def test__load_data_with_invalid_path(self):
        self.dc.bids = ['bid-eng-DBY', 'bid-fra-fob']
        self.dc.sd_path_prefix = 'test/data/invalid_path'

        with self.assertRaises(FileNotFoundError):
            self.dc._load_data()

    # def test__build_sds(self):
    #     self.fail()

    def test__lemmatize_verse(self):
        result = self.dc._lemmatize_english_verse(['I', 'drank', 'water', '.'])
        self.assertEqual(result, ['I', 'drink', 'water', '.'])

    # def test__tokenize_verses(self):
    #     self.fail()

    # def test__add_directed_edge(self):
    #     self.fail()

    # def test__map_word_to_qid(self):
    #     self.fail()

    # def test__map_word_to_qid_bidirectionally(self):
    #     self.fail()

    def test__map_two_bibles(self):
        self.dc.verses_by_bid = {
            'bid-eng-DBY-10': [
                'In the beginning...\n',
                'And the earth...\n',
                '\n',
                '\n',
            ],
            'bid-fra-fob-10': [
                'Au commencement...',
                '\n',
                '\n',
                '\n',
            ]
        }
        eng_word_1 = self._create_word('in', 'eng')
        eng_word_2 = self._create_word('the', 'eng')
        eng_word_3 = self._create_word('beginning', 'eng')
        eng_word_4 = self._create_word('...', 'eng')
        eng_word_5 = self._create_word('and', 'eng')
        eng_word_6 = self._create_word('earth', 'eng', {'1.2 World'})
        fra_word_1 = self._create_word('au', 'fra')
        fra_word_2 = self._create_word('commencement', 'fra', {'8.4.6.1.1 Débuter'})
        fra_word_3 = self._create_word('...', 'fra')
        self.dc.words_by_text_by_lang = {
            'eng': {
                'in': eng_word_1,
                'the': eng_word_2,
                'beginning': eng_word_3,
                '...': eng_word_4,
                'and': eng_word_5,
                'earth': eng_word_6,
            },
            'fra': {
                'au': fra_word_1,
                'commencement': fra_word_2,
                '...': fra_word_3,
            }
        }
        self.dc.wtxts_by_verse_by_bid = {
            'bid-eng-DBY-10': [
                ['in', 'the', 'beginning', '...'],
                ['and', 'the', 'earth', '...'],
                [],
                [],
            ],
            'bid-fra-fob-10': [
                ['au', 'commencement', '...'],
                [],
                [],
                [],
            ],
        }
        alignment = [
            '0-0 1-0 2-1 3-2\n',
            '0-0\n',
            '0-0\n',
            '0-0\n',
        ]

        self.dc._map_two_bibles_bidirectionally(alignment, 'bid-eng-DBY-10', 'bid-fra-fob-10')

        self.assertListEqual([(eng_word_1, 1), (eng_word_2, 1)],
                             list(fra_word_1.get_aligned_words_and_counts(self.dc.words_by_text_by_lang)))
        self.assertDictEqual({
            'eng': {
                'fra': {
                    '8.4.6.1.1 Débuter': ', beginning'
                }
            },
            'fra': {
                'eng': {}
            }
        }, self.dc.aligned_wtxts_by_qid_by_lang_by_lang)

    # def test__map_words_to_qids(self):
    #     self.fail()

    def test__build_word_graph(self):
        self._initialize_5_german_words()

        self.dc._build_word_graph()

        self.assertEqual(self.dc.word_graph.number_of_nodes(), 10)
        self.assertEqual(self.dc.word_graph.number_of_edges(), 9)

    def test__build_word_graph_with_additional_languages(self):
        # load languages that do not appear in the target languages
        words = self._initialize_words_for_3_languages()
        self.dc._build_word_graph()

        # check if word_graph contains a certain node
        [self.assertTrue(self.dc.word_graph.has_node(word)) for word in words['eng']]
        [self.assertTrue(self.dc.word_graph.has_node(word)) for word in words['fra']]
        [self.assertFalse(self.dc.word_graph.has_node(word)) for word in words['deu']]
        self.assertEqual(self.dc.word_graph.number_of_nodes(), 6)
        self.assertEqual(self.dc.word_graph.number_of_edges(), 7)

    # def test_plot_subgraph(self):
    #     self.fail()

    # def test__find_lemma_link_candidates(self):
    #     self.fail()

    # def test__find_translation_link_candidates(self):
    #     self.fail()

    # def test__compute_sum_of_weights(self):
    #     self.fail()

    # def test__compute_link_score(self):
    #     self.fail()

    def _initialize_4_german_words(self):
        self.dc.target_langs = ['eng', 'deu']

        eng_word_1_1 = self._create_word('pass each other', 'eng', None, 30)
        eng_word_1_2 = self._create_word('passed each other', 'eng', None, 20)
        eng_word_1_3 = self._create_word('passing each other', 'eng', None, 10)

        deu_word_1_1 = self._create_word('aneinander vorbeigehen', 'deu', None, 40)
        deu_word_1_2 = self._create_word('aneinander vorbeigegangen', 'deu', None, 30)
        deu_word_1_3 = self._create_word('aneinander vorbeigehend', 'deu', None, 20)
        deu_word_1_4 = self._create_word('aneinander vorbeigingen', 'deu', None, 10)

        # correct edges
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_1, 10)
        self.dc._add_bidirectional_edge(eng_word_1_2, deu_word_1_2, 10)
        self.dc._add_bidirectional_edge(eng_word_1_3, deu_word_1_3, 10)
        self.dc._add_bidirectional_edge(eng_word_1_2, deu_word_1_4, 10)

        # noisy edges
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_2, 5)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_3, 5)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_4, 5)

    def _initialize_5_german_words(self):
        self.dc.target_langs = ['eng', 'deu']

        eng_word_1_1 = self._create_word('pass', 'eng',
                                         {'4.2.6.2.1 Football, soccer 9', '4.7.2 Pass laws 1',
                                          '3.6.7 Test 2'}, 30)
        eng_word_1_2 = self._create_word('passed', 'eng', None, 20)
        eng_word_1_3 = self._create_word('passing', 'eng', {'3.4 Emotion 4', '3.6.7 Test 2'}, 10)
        eng_word_2_1 = self._create_word('human being', 'eng', {'2 Person 1'}, 50)
        eng_word_2_2 = self._create_word('human beings', 'eng', {'2 Person 1'}, 40)

        deu_word_1_1 = self._create_word('vorbeigehen', 'deu', None, 30)
        deu_word_1_2 = self._create_word('vorbeigegangen', 'deu', None, 20)
        deu_word_1_3 = self._create_word('vorbeigehend', 'deu', None, 10)
        deu_word_2_1 = self._create_word('mensch', 'deu', None, 50)
        deu_word_2_2 = self._create_word('menschen', 'deu', None, 40)

        # correct edges
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_1, 10)
        self.dc._add_bidirectional_edge(eng_word_1_2, deu_word_1_2, 10)
        self.dc._add_bidirectional_edge(eng_word_1_3, deu_word_1_3, 10)
        self.dc._add_bidirectional_edge(eng_word_2_1, deu_word_2_1, 10)
        self.dc._add_bidirectional_edge(eng_word_2_2, deu_word_2_2, 10)

        # noisy edges
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_2, 5)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_1_3, 5)
        self.dc._add_bidirectional_edge(eng_word_1_1, deu_word_2_2, 1)
        self.dc._add_bidirectional_edge(eng_word_2_1, deu_word_2_2, 5)

    def test__predict_lemma_links(self):
        # designed to cover unlikely branches
        self._initialize_4_german_words()

        self.dc._build_word_graph()
        self.dc._plot_subgraph('eng', 'pass each other')
        lemma_link_candidates = [
            (self.dc.words_by_text_by_lang['deu']['aneinander vorbeigingen'],
             self.dc.words_by_text_by_lang['deu']['aneinander vorbeigegangen']),

            (self.dc.words_by_text_by_lang['deu']['aneinander vorbeigehend'],
             self.dc.words_by_text_by_lang['deu']['aneinander vorbeigehen']),

            (self.dc.words_by_text_by_lang['deu']['aneinander vorbeigingen'],
             self.dc.words_by_text_by_lang['deu']['aneinander vorbeigehend']),
        ]

        self.dc._predict_lemma_links(lemma_link_candidates)

        self.assertEqual({
            'deu': {
                'aneinander vorbeigegangen': 'aneinander vorbeigehen',
                'aneinander vorbeigehen': 'aneinander vorbeigehen',
                'aneinander vorbeigehend': 'aneinander vorbeigehen',
                'aneinander vorbeigingen': 'aneinander vorbeigehen',
            }
        }, self.dc.base_lemma_by_wtxt_by_lang)
        self.assertEqual({
            'deu': {
                'aneinander vorbeigehen': {
                    'aneinander vorbeigehen',
                    'aneinander vorbeigegangen',
                    'aneinander vorbeigehend',
                    'aneinander vorbeigingen',
                },
            }
        }, self.dc.lemma_group_by_base_lemma_by_lang)

    def test__predict_lemmas(self):
        self._initialize_5_german_words()

        self.dc._build_word_graph()
        self.dc._plot_subgraph('eng', 'pass')
        self.dc._predict_lemmas()

        self.assertEqual({
            'eng': {
                'human being': 'human being',
                'human beings': 'human being',
            },
            'deu': {
                'vorbeigehen': 'vorbeigehen',
                'vorbeigegangen': 'vorbeigehen',
                'vorbeigehend': 'vorbeigehen',
                'mensch': 'mensch',
                'menschen': 'mensch',
            }
        }, self.dc.base_lemma_by_wtxt_by_lang)
        self.assertEqual({
            'eng': {
                'human being':
                    {'human being', 'human beings'}
            },
            'deu': {
                'vorbeigehen': {'vorbeigehen', 'vorbeigegangen', 'vorbeigehend'},
                'mensch': {'mensch', 'menschen'}
            }
        }, self.dc.lemma_group_by_base_lemma_by_lang)

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
        self.dc._build_word_graph()
        self.dc._plot_subgraph('fra', 'boire')

        self.assertEqual(self.dc.words_by_text_by_lang['fra'].keys(), {'boire', 'eau'})
        self.assertEqual(self.dc.words_by_text_by_lang['fra']['boire'].display_text, 'BOIRE (3)')
        self.assertEqual(self.dc.words_by_text_by_lang['fra']['eau'].display_text, 'EAU (2)')

    def test_predict_links_in_2_languages(self):
        self._initialize_5_german_words()

        self.dc._build_word_graph()
        self.dc._predict_translation_links()
        self.dc._plot_subgraph('eng', 'pass')

        self.assertDictEqual({
            'eng': {},
            'deu': {
                '2 Person 1': {
                    'mensch': (0.8, 'human being'),
                    'menschen': (0.7692307692307693, 'human beings'),
                },
                '3.4 Emotion 4': {
                    'vorbeigehend': (0.8, 'passing')
                },
                '3.6.7 Test 2': {
                    'menschen': (0.05405405405405406, 'pass'),
                    'vorbeigegangen': (0.2777777777777778, 'pass'),
                    'vorbeigehen': (0.6451612903225806, 'pass'),
                    'vorbeigehend': (0.8, 'passing'),
                },
                '4.2.6.2.1 Football, soccer 9': {
                    'menschen': (0.05405405405405406, 'pass'),
                    'vorbeigegangen': (0.2777777777777778, 'pass'),
                    'vorbeigehen': (0.6451612903225806, 'pass'),
                    'vorbeigehend': (0.2777777777777778, 'pass'),
                },
                '4.7.2 Pass laws 1': {
                    'menschen': (0.05405405405405406, 'pass'),
                    'vorbeigegangen': (0.2777777777777778, 'pass'),
                    'vorbeigehen': (0.6451612903225806, 'pass'),
                    'vorbeigehend': (0.2777777777777778, 'pass'),
                }}
        }, dict(self.dc.top_scores_by_qid_by_lang))

    def _initialize_words_for_3_languages(self):
        eng_word_1 = self._create_word('drink', 'eng', {'5.2.2.7 Drink 1', '5.2.3.6 Beverage 1'}, 3)
        eng_word_2 = self._create_word('water', 'eng', {'1.2.3 Solid, liquid, gas 2'}, 2)
        eng_word_3 = self._create_word('the', 'eng', {'9.2.3.5 Demonstrative pronouns 1'}, 100)
        fra_word_1 = self._create_word('boire', 'fra', None, 3)
        fra_word_2 = self._create_word('eau', 'fra', None, 5)
        fra_word_3 = self._create_word('les', 'fra', None, 100)
        deu_word_1 = self._create_word('trinken', 'deu', None, 3)
        deu_word_2 = self._create_word('wasser', 'deu', None, 2)
        deu_word_3 = self._create_word('die', 'deu', None, 100)

        self.dc._add_bidirectional_edge(eng_word_1, fra_word_1, 10)
        self.dc._add_bidirectional_edge(eng_word_1, deu_word_1, 10)
        self.dc._add_bidirectional_edge(fra_word_1, deu_word_1, 10)
        self.dc._add_bidirectional_edge(eng_word_2, fra_word_2, 10)
        self.dc._add_bidirectional_edge(eng_word_2, deu_word_2, 10)
        self.dc._add_bidirectional_edge(fra_word_2, deu_word_2, 10)
        self.dc._add_bidirectional_edge(eng_word_3, fra_word_3, 10)
        self.dc._add_bidirectional_edge(eng_word_3, deu_word_3, 10)
        self.dc._add_bidirectional_edge(fra_word_3, deu_word_3, 10)

        # add some noise
        self.dc._add_bidirectional_edge(eng_word_1, fra_word_3, 2)
        self.dc._add_bidirectional_edge(eng_word_1, deu_word_3, 2)
        self.dc._add_bidirectional_edge(eng_word_2, fra_word_3, 2)
        self.dc._add_bidirectional_edge(eng_word_2, deu_word_3, 2)
        self.dc._add_bidirectional_edge(fra_word_1, eng_word_3, 2)
        self.dc._add_bidirectional_edge(fra_word_1, deu_word_3, 2)
        self.dc._add_bidirectional_edge(fra_word_2, eng_word_3, 2)
        self.dc._add_bidirectional_edge(fra_word_2, deu_word_3, 2)
        self.dc._add_bidirectional_edge(deu_word_1, fra_word_3, 2)
        self.dc._add_bidirectional_edge(deu_word_1, eng_word_3, 2)
        self.dc._add_bidirectional_edge(deu_word_2, fra_word_3, 2)
        self.dc._add_bidirectional_edge(deu_word_2, eng_word_3, 2)

        return {
            'eng': [eng_word_1, eng_word_2, eng_word_3],
            'fra': [fra_word_1, fra_word_2, fra_word_3],
            'deu': [deu_word_1, deu_word_2, deu_word_3]
        }

    def test_predict_links_in_3_languages(self):
        self.dc.target_langs = ['eng', 'fra', 'deu']
        self._initialize_words_for_3_languages()

        self.dc._build_word_graph()
        self.dc._predict_translation_links()
        self.dc._plot_subgraph('eng', 'the')

        self.assertDictEqual({
            'eng': {},
            'deu': {
                '1.2.3 Solid, liquid, gas 2': {
                    'wasser': (0.8333333333333334, 'water'),
                    'die': (0.15384615384615385, 'water'),
                },
                '5.2.2.7 Drink 1': {
                    'trinken': (0.8333333333333334, 'drink'),
                    'die': (0.15384615384615385, 'drink'),
                },
                '5.2.3.6 Beverage 1': {
                    'trinken': (0.8333333333333334, 'drink'),
                    'die': (0.15384615384615385, 'drink'),
                },
                '9.2.3.5 Demonstrative pronouns 1': {
                    'die': (0.7142857142857143, 'the'),
                    'trinken': (0.15384615384615385, 'the'),
                    'wasser': (0.15384615384615385, 'the'),
                }},
            'fra': {
                '1.2.3 Solid, liquid, gas 2': {
                    'eau': (0.8333333333333334, 'water'),
                    'les': (0.15384615384615385, 'water'),
                },
                '5.2.2.7 Drink 1': {
                    'boire': (0.8333333333333334, 'drink'),
                    'les': (0.15384615384615385, 'drink'),
                },
                '5.2.3.6 Beverage 1': {
                    'boire': (0.8333333333333334, 'drink'),
                    'les': (0.15384615384615385, 'drink'),
                },
                '9.2.3.5 Demonstrative pronouns 1': {
                    'les': (0.7142857142857143, 'the'),
                    'boire': (0.15384615384615385, 'the'),
                    'eau': (0.15384615384615385, 'the'),
                }}
        }, dict(self.dc.top_scores_by_qid_by_lang))

    # def test__filter_target_sds_with_threshold(self):
    #     self.fail()

    # def test__compute_f1_score(self):
    #     self.fail()

    def test__compute_f1_score_without_target_semantic_domains(self):
        result = self.dc._compute_f1_score(None, 'deu')
        self.assertEqual((None, None, None, None, None), result)

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

    def test__compute_mean_reciprocal_rank_without_target_semantic_domains(self):
        result = self.dc._compute_mean_reciprocal_rank('urd')
        self.assertEqual(None, result)

    # def test__evaluate(self):
    #     self.fail()

    def test_evaluate_without_source_semantic_domains(self):
        self.dc._evaluate()
        self.assertDictEqual({}, self.dc.evaluation_results_by_lang)

    def test_no_progress_loaded(self):
        with self.assertRaises(AssertionError):
            self.dc._execute_and_track_state(self.dc._build_word_graph, step_name='_build_word_graph (raw)',
                                             load=False, save=True)

    def test_inconsistent_state_loaded(self):
        # only word_graph has been loaded
        self.dc.progress_log = [
            '_preprocess_data',
            '_map_words_to_qids',
            '_build_word_graph (raw)',
        ]
        self.dc.word_graph = nx.Graph()
        self.dc._execute_and_track_state(self.dc._preprocess_data, load=False, save=True)
        with self.assertRaises(AssertionError):
            self.dc._execute_and_track_state(self.dc._map_words_to_qids, load=False, save=True)

    def test_inconsistent_step_order(self):
        # the step '_map_words_to_qids' is missing
        self.dc.progress_log = [
            '_preprocess_data',
            '_build_word_graph (raw)',
        ]
        self.dc.word_graph = nx.Graph()
        with self.assertRaises(AssertionError):
            self.dc._execute_and_track_state(self.dc._preprocess_data, load=False, save=True)


class TestLinkPredictionDictionaryCreatorSlow(TestLinkPredictionDictionaryCreator):
    # This class is for slower test cases.
    def test_full_pipeline_with_loading_and_with_link_prediction(self):
        self._run_full_pipeline_twice(load_1=True, save_1=True, load_2=True, save_2=True,
                                      plot_word_lang='fra', plot_wtxt='et', min_count=2)

    def test_full_pipeline_without_loading_and_with_all_sds_and_with_link_prediction(self):
        self._run_full_pipeline_twice(check_isomorphism=False, load_1=False, save_1=False, load_2=False, save_2=False,
                                      sd_path_prefix='../semdom extractor/output/semdom_qa_clean')

    def test__combine_alignments(self):
        self.dc.wtxts_by_verse_by_bid = {
            'bid-eng-DBY-10': [
                ['in', 'the', 'beginning', 'god', 'create', 'the', 'heaven', 'and', 'the', 'earth', '.'],
                ['and', 'the', 'earth', 'be', 'waste', 'and', 'empty', ',', 'and', 'darkness', 'be', 'on', 'the',
                 'face', 'of', 'the', 'deep', ',', 'and', 'the', 'spirit', 'of', 'god', 'be', 'hover', 'over', 'the',
                 'face', 'of', 'the', 'water', '.'],
                [],
                [],
            ],
            'bid-fra-fob-10': [
                ['au', 'commencement', ',', 'dieu', 'créa', 'les', 'cieux', 'et', 'la', 'terre', '.'],
                ['or', 'la', 'terre', 'était', 'informe', 'et', 'vide', ',', 'et', 'les', 'ténèbres', 'étaient',
                 'à', 'la', 'surface', 'de', 'l', "'", 'abîme', ',', 'et', 'l', "'", 'esprit', 'de', 'dieu', 'se',
                 'mouvait', 'sur', 'les', 'eaux', '.'],
                [],
                [],
            ],
        }
        self.dc.num_verses = 4

        self.dc._combine_alignments()

        self.assertTrue(os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_diag.align') or
                        os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_awesome.align'))
        self.assertTrue(os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-fra-fob-10_bpe_diag.align') or
                        os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_awesome.align'))

    def test__combine_alignments_with_missing_verse(self):
        self.dc.wtxts_by_verse_by_bid = {
            'bid-eng-DBY-10': [
                ['in', 'the', 'beginning', 'god', 'create', 'the', 'heaven', 'and', 'the', 'earth', '.'],
                ['and', 'the', 'earth', 'be', 'waste', 'and', 'empty', ',', 'and', 'darkness', 'be', 'on', 'the',
                 'face', 'of', 'the', 'deep', ',', 'and', 'the', 'spirit', 'of', 'god', 'be', 'hover', 'over', 'the',
                 'face', 'of', 'the', 'water', '.'],
                [],
                [],
            ],
            'bid-fra-fob-10': [
                ['au', 'commencement', ',', 'dieu', 'créa', 'les', 'cieux', 'et', 'la', 'terre', '.'],
                [],
                [],
                [],
            ],
        }
        self.dc.num_verses = 4

        self.dc._combine_alignments()

        self.assertTrue(os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_diag.align') or
                        os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_awesome.align'))
        self.assertTrue(os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-fra-fob-10_bpe_diag.align') or
                        os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_awesome.align'))

    def test__preprocess_data(self):
        self.dc._preprocess_data()

        self.assertTrue(os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_diag.align') or
                        os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_awesome.align'))
        self.assertTrue(os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-fra-fob-10_bpe_diag.align') or
                        os.path.isfile('test/data/1_aligned_bibles/bid-eng-DBY-10_bid-eng-DBY-10_bpe_awesome.align'))
