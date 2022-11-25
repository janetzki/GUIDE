from src.dictionary_creator.tfidf_dictionary_creator import TfidfDictionaryCreator
from test.dictionary_creator.test_dictionary_creator import AbstractTestDictionaryCreator


class TestTfidfDictionaryCreator(AbstractTestDictionaryCreator):
    def setUp(self) -> None:
        super().setUp()
        self.tested_class = TfidfDictionaryCreator
        self.dc = self._create_dictionary_creator()

    def test_full_pipeline_with_loading_and_with_tfidf(self):
        self._run_full_pipeline_twice(load_1=False, save_1=True, load_2=True, save_2=False,
                                      plot_word_lang='fra', plot_word='et', min_count=3)

    def test__train_tfidf_based_model(self):
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

        self.dc._train_tfidf_based_model()

        self.assertEqual({
            'eng': {'1.2 1': {'earth': 1.0},
                    '4.9.6 1': {'and': 0.447213595499958,
                                'heaven': 0.894427190999916}},
            'fra': {'1.2.3 1': {'avec': 0.10540925533894598,
                                'eaux': 0.8432740427115678,
                                'les': 0.5270462766947299},
                    '8.3.3.2 6': {'ténèbres': 1.0}}
        }, self.dc.top_scores_by_qid_by_lang)
