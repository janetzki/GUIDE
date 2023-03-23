from unittest import TestCase

from src.dictionary_creator.tfidf_dictionary_creator import TfidfDictionaryCreator
from src.semantic_domain_identifier import SemanticDomainIdentifier


class TestSemanticDomainIdentifier(TestCase):
    def setUp(self) -> None:
        dc = TfidfDictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
        dc.top_scores_by_qid_by_lang = {
            'eng': {
                '1.1.1.1 1': {
                    'moon': (0.2, 'croissant'),
                    'lunar': (0.3, 'croissant'),
                },
                '1.1.1.2 1': {
                    'star': (0.4, 'croissant'),
                    'stellar': (0.5, 'croissant'),
                },
            },
        }
        dc.question_by_qid_by_lang = {
            'eng': {
                '1.1.1.1 1': 'What words refer to the moon?',
                '1.1.1.2 1': 'What words are used to refer to the stars?',
            },
        }
        self.sdi = SemanticDomainIdentifier(dc)
        self.phrases = [
            ['the', 'moon', 'is', 'a', 'huge', 'croissant', '.'],
            ['this', 'star', 'is', 'shining', 'brightly', '.'],
        ]

    def test_identify_semantic_domains(self):
        qids = self.sdi.identify_semantic_domains(self.phrases)
        self.assertTrue(type(qids) == list)

    def test_identify_semantic_domains_not_empty(self):
        qids = self.sdi.identify_semantic_domains(self.phrases)

        self.assertTrue(len(qids))
