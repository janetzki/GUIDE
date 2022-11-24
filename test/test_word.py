from unittest import TestCase

from src.word import Word


class TestWord(TestCase):
    def setUp(self) -> None:
        self.word = Word('drink', 'eng', occurrences_in_bible=1)

    # def test_get_aligned_words_and_counts(self):
    #     self.fail()

    def test_get_aligned_words_and_counts_empty(self):
        result = list(self.word.get_aligned_words_and_counts({}))
        self.assertEqual([], result)

    # def test_add_aligned_word(self):
    #     self.fail()

    # def test_remove_alignment(self):
    #     self.fail()

    # def test_merge_words(self):
    #     # todo: assert that the aligned words stay symmetric
    #     self.fail()

    def test___repr__(self):
        self.assertEqual(repr(self.word), 'eng: drink')
