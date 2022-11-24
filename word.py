from collections import Counter


class Word(object):
    def __init__(self, text, lang, qids=None, occurrences_in_bible=0):
        self.text = text
        self.iso_language = lang
        assert (qids is None or type(qids) == set)
        self.qids = set() if qids is None else qids
        self.occurrences_in_bible = occurrences_in_bible
        self.display_text = text
        self._aligned_words = Counter()

    def __eq__(self, other):
        return type(other) == Word \
               and self.text == other.text \
               and self.iso_language == other.iso_language \
               and self.qids == other.qids \
               and self.occurrences_in_bible == other.occurrences_in_bible \
               and self.display_text == other.display_text \
               and self._aligned_words == other._aligned_words

    def __hash__(self):
        return hash((self.iso_language, self.text))

    def __str__(self):
        return f'{self.iso_language}: {self.text}'

    def __repr__(self):
        return f'{self.iso_language}: {self.text}'

    def get_aligned_words_and_counts(self, words_by_text_by_lang):
        for word_str, count in self._aligned_words.items():
            lang, wtxt = word_str.split(': ')
            word = words_by_text_by_lang[lang][wtxt]
            yield word, count

    def add_aligned_word(self, word, count=1):
        # caution: this is not symmetric, todo: make this symmetric
        if self.text == "'" and self.iso_language == 'fra' and word.text == 'the':
            a = 3
        self._aligned_words[str(word)] += count

    def remove_alignment(self, word):
        del self._aligned_words[str(word)]

    def _update_aligned_words(self, lang, removed_word, words_by_text_by_lang):
        # replace references to a removed word (e.g., drink, drank, drunk)
        # with references to a merged word (e.g., DRINK)
        for aligned_word, count in words_by_text_by_lang[lang][removed_word.text]. \
                get_aligned_words_and_counts(words_by_text_by_lang):
            aligned_word.remove_alignment(removed_word)
            aligned_word.add_aligned_word(self, count)
        del words_by_text_by_lang[lang][removed_word.text]

    def merge_words(self, words, words_by_text_by_lang, strength_by_lang_by_wtxt_by_lang, changed_variables):
        # todo: refactor this method by creating a new node instead of modifying an existing one --> call _update_aligned_words only once
        self.display_text = f'{self.text.upper()} ({len(words) + 1})'
        print(self.display_text)
        for word in words:
            print(word)
            self.qids.update(
                word.qids)  # todo: weight qids by occurrences in Bible when adding semdoms as nodes to graph
            self.occurrences_in_bible += word.occurrences_in_bible
            self._aligned_words += word._aligned_words
            self._update_aligned_words(word.iso_language, word, words_by_text_by_lang)
            strength_by_lang_by_wtxt_by_lang[word.iso_language].pop(word.text, None)  # remove cache entry
        print('\n')
        self._update_aligned_words(self.iso_language, self, words_by_text_by_lang)
        strength_by_lang_by_wtxt_by_lang[self.iso_language].pop(self.text, None)  # remove cache entry
        changed_variables.add('words_by_text_by_lang')
        changed_variables.add('strength_by_lang_by_wtxt_by_lang')
