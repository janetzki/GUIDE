import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.dictionary_creator.dictionary_creator import DictionaryCreator


class TfidfDictionaryCreator(DictionaryCreator):
    STEPS = [
        '_preprocess_data',
        '_map_words_to_qids',
        '_remove_stop_words',
        '_train_tfidf_based_model',
        '_evaluate',
    ]

    LOADED_VARIABLES_BY_STEP = DictionaryCreator.LOADED_VARIABLES_BY_STEP | {
        '_train_tfidf_based_model': ['top_scores_by_qid_by_lang'],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()

    def _train_tfidf_based_model(self):
        for target_lang in self.target_langs:
            aligned_wtxts_by_qid = self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][self.source_lang]
            tfidfs = self.vectorizer.fit_transform(list(aligned_wtxts_by_qid.values()))
            assert tfidfs.shape[0] == len(aligned_wtxts_by_qid)
            for idx, tfidf in tqdm(enumerate(tfidfs),
                                   desc=f'Collecting top {target_lang} tf-idf scores',
                                   total=tfidfs.shape[
                                       0]):  # caution: might fail in debug mode with Python 3.10 instead of Python 3.9
                qid = list(aligned_wtxts_by_qid.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=['TF-IDF'])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                df = df.head(20)

                # convert df to dict
                scores_by_wtxt = {word: (score, None) for word, score in
                                  zip(df.index, df['TF-IDF'])}

                self.top_scores_by_qid_by_lang[target_lang][qid] = scores_by_wtxt

    def create_dictionary(self, load=False, save=False):
        self._execute_and_track_state(self._preprocess_data, load=load, save=save)
        self._execute_and_track_state(self._map_words_to_qids, load=load, save=save)
        self._execute_and_track_state(self._remove_stop_words, load=load, save=save)
        self._execute_and_track_state(self._train_tfidf_based_model, load=load, save=save)
        self._execute_and_track_state(self._evaluate, print_reciprocal_ranks=False)
