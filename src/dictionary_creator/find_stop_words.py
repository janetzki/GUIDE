from collections import defaultdict, Counter

from nltk.corpus import stopwords
from tqdm import tqdm


def find_stop_words(dc):
    # https://gist.github.com/sebleier/554280
    stop_words = defaultdict(set)
    stop_words['eng'] = set(stopwords.words('english'))

    # remove all stop words that do not occur in dc.words_by_text_by_lang['eng']
    stop_words['eng'] = set([w for w in stop_words['eng'] if w in dc.words_by_text_by_lang['eng']])

    aligned_words_by_lang = defaultdict(Counter)
    eng_word_occurrences = Counter()

    # add aligned words to stop words
    for wtxt in tqdm(stop_words['eng'], desc='Finding stop words...', total=len(stop_words['eng'])):
        word = dc.words_by_text_by_lang['eng'][wtxt]
        aligned_words = dict(word.get_aligned_words_and_counts(dc.words_by_text_by_lang))
        if len(aligned_words) == 0:
            continue
        eng_word_occurrences[wtxt] = aligned_words[word]

        # predicted_links = [(w, dc._compute_link_score(word, w)) for w, count in aligned_words.items()]
        # predicted_links = [(w, link_score) for w, link_score in predicted_links if link_score >= dc.score_threshold]
        #
        # for w, link_score in predicted_links:
        #     if w.iso_language != 'eng':
        #         stop_words[w.iso_language].add(w.text)

        for w, count in aligned_words.items():
            if w.iso_language != 'eng':
                aligned_words_by_lang[w.iso_language][w.text] += count

    total_eng_word_occurrences = sum(eng_word_occurrences.values())

    # sort aligned words by count, filter them, and add them to the stop words
    threshold = max(10, total_eng_word_occurrences / 10000)
    for lang in aligned_words_by_lang:
        aligned_words_by_lang[lang] = Counter(
            {w: c for w, c in aligned_words_by_lang[lang].most_common() if c >= threshold})
        stop_words[lang].update(aligned_words_by_lang[lang].keys())

    return stop_words
