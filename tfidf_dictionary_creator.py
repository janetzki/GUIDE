import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from dictionary_creator import DictionaryCreator


class TfidfDictionaryCreator(DictionaryCreator):
    STEPS = [
        'started',
        'preprocessed data',
        'mapped words to qids',
        'built uncontracted word graph',
        'predicted lemmas',
        'contracted lemmas',
        'built contracted word graph',
        'predicted translation links',
        'trained tfidf based model',
        'evaluated',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()

    def train_tfidf_based_model(self):
        for target_lang in self.target_langs:
            if target_lang in self.top_scores_by_qid_by_lang:
                print(f'Skipped: top {target_lang} tfidfs already collected')
                continue

            aligned_wtxts_by_qid = self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][self.source_lang]
            tfidfs = self.vectorizer.fit_transform(list(aligned_wtxts_by_qid.values()))
            assert (tfidfs.shape[0] == len(aligned_wtxts_by_qid))
            for idx, tfidf in tqdm(enumerate(tfidfs),
                                   desc=f'Collecting top {target_lang} tf-idf scores',
                                   total=tfidfs.shape[0]):
                qid = list(aligned_wtxts_by_qid.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=['TF-IDF'])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                df = df.head(20)

                # convert df to dict
                scores_by_wtxt = {word: score for word, score in zip(df.index, df['TF-IDF'])}

                self.top_scores_by_qid_by_lang[target_lang][qid] = scores_by_wtxt
                self.changed_variables.add('top_scores_by_qid_by_lang')

    def create_dictionary(self, load=False, save=False, plot_word_lang='eng', plot_wtxt='drink', min_count=1):
        self.execute_and_track_state(self.preprocess_data, load=load, save=save)
        self.execute_and_track_state(self.map_words_to_qids, load=load, save=save)
        self.execute_and_track_state(self.build_word_graph, step_name='build uncontracted word graph',
                                     load=load, save=save)
        self.execute_and_track_state(self.train_tfidf_based_model, load=load, save=save)
        self.execute_and_track_state(self.evaluate, print_reciprocal_ranks=False)