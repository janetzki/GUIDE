from unittest import TestCase

from src.semantic_domain_identifier import SemanticDomainIdentifier


class TestSemanticDomainIdentifier(TestCase):
    def setUp(self) -> None:
        self.sdi = SemanticDomainIdentifier()

    def test_identify_semantic_domains(self):
        qids = self.sdi.identify_semantic_domains()
        self.assertTrue(len(qids))
