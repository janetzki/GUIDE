import json
import os
import pickle
from collections import defaultdict

import pandas as pd
import spacy
from tqdm import tqdm

from src.dictionary_creator.dictionary_creator import DictionaryCreator


class AlignmentFileCreator(DictionaryCreator):
    def __init__(self, bids):
        super().__init__(bids)
        self.answered_questions = pd.read_csv(
            'data/2_sd_labeling/matched_questions_gospels_raw_answered_positive.csv')
        self.pharaoh_alignments_by_bid_by_bid_by_verse_id = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    # self.verses_by_bid[bid][:23213] = [''] * 23213  # start at NT
    # self.verses_by_bid[bid][23214:] = [''] * len(self.verses_by_bid[bid][23214:])
    # self.verses_by_bid[bid][26992:] = [''] * len(self.verses_by_bid[bid][26992:])  # all 4 gospels
    # self.verses_by_bid[bid][3:] = [''] * (self.num_verses - 3)

    # if idx < 31170:
    #     # ignore apocrypha
    #     bible_reference = convert_verse_id_to_bible_reference(idx)

    # if idx not in range(23213, 26992): # all 4 gospels # not in range(0, 3):
    #     continue

    #  long_verse_id = str(self.verse_ids[idx])
    # self.pharaoh_alignments_by_bid_by_bid_by_verse_id[long_verse_id][bid_1][
    #                     bid_2] += f'{wtxt_1_idx}-{wtxt_2_idx} '
    #                 if bid_1 != bid_2:
    #                     self.pharaoh_alignments_by_bid_by_bid_by_verse_id[long_verse_id][bid_2][
    #                         bid_1] += f'{wtxt_2_idx}-{wtxt_1_idx} '

    def _label_eng_tokens(self, verse_id, tokens):
        qids = ['X'] * len(tokens)

        qid_df = self.answered_questions[self.answered_questions.verse_id == verse_id]
        for idx, row in qid_df.iterrows():
            qid_tokens = row['tokens'].split(' ')
            for token in qid_tokens:
                # assign qid to all tokens that match
                for i, t in enumerate(tokens):
                    if t == token:
                        qids[i] = row['qid']  # TODO: handle same token multiple times in one verse (match context)
        return qids

    def _label_fra_tokens(self, verse_id, tokens):
        # Heuristic: If a token belongs to a QID that also belongs to the English verse, label it with that QID.
        qids = ['X'] * len(tokens)

        qid_df = self.answered_questions[self.answered_questions.verse_id == verse_id]
        all_eng_qids = set()
        for idx, row in qid_df.iterrows():
            all_eng_qids.add(row['qid'])

        lemmas = self.fra_lemmas_by_verse[verse_id]
        for idx, lemma in enumerate(lemmas):
            if lemma not in self.gt_qids_by_wtxt:
                continue
            for qid in self.gt_qids_by_wtxt[lemma]:
                if qid in all_eng_qids:
                    qids[idx] = qid

        for idx, token in enumerate(tokens):
            if token not in self.gt_qids_by_wtxt:
                continue
            for qid in self.gt_qids_by_wtxt[token]:
                if qid in all_eng_qids:
                    qids[idx] = qid

        return qids

    def _lemmatize_fra_verses(self, verses):
        nlp = spacy.load('fr_core_news_md', disable=['ner', 'parser'])
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))
        lemmas_by_verse = []
        for doc in nlp.pipe([' '.join(tokens) for tokens in verses]):
            lemmas = [token.lemma_ for token in doc]
            lemmas_by_verse.append(['de' if lemma == 'd' else lemma for lemma in lemmas])
        return lemmas_by_verse

    def _label_tokens_with_gt_qids(self, bid, verse_id, tokens):
        """
        Label each token with its ground-truth QID.
        """
        lang = self._convert_bid_to_lang(bid)
        if len(tokens) == 0:
            return []
        if lang == 'eng':
            return self._label_eng_tokens(verse_id, tokens)
        if lang == 'fra':
            return self._label_fra_tokens(verse_id, tokens)
        return ['X'] * len(tokens)

    def _save_bible_to_conllu_file(self, bid):
        # build a ground truth dictionary that maps wtxts to qids
        self.gt_qids_by_wtxt = defaultdict(set)
        if 'fra' in self.words_by_text_by_lang:
            for word in self.words_by_text_by_lang['fra'].values():
                for qid in word.qids:
                    self.gt_qids_by_wtxt[word.text].add(qid)
        self.gt_qids_by_wtxt = dict(self.gt_qids_by_wtxt)

        if self._convert_bid_to_lang(bid) == 'fra':
            # TODO: Do this before tokenization (e.g., for "d'" -> "de" instead of "d" --> "de")
            self.fra_lemmas_by_verse = self._lemmatize_fra_verses(self.wtxts_by_verse_by_bid[bid])

        file_path = os.path.join('../external_repos/GNN-POSTAG/dataset/gdfa_final/tokenized_bibles/',
                                 f'{bid}_sd.conllu')
        with open(file_path, 'w') as conllu_file:
            # for each verse_id, write each verse's tokens to the conllu file
            for verse_id, tokens in tqdm(enumerate(self.wtxts_by_verse_by_bid[bid]),
                                         desc=f'Writing {bid} to {file_path}',
                                         total=len(self.wtxts_by_verse_by_bid[bid])):
                long_verse_id = self.verse_ids[verse_id]
                conllu_file.write(f'# verse_id: {long_verse_id}\n')
                sds = self._label_tokens_with_gt_qids(bid, verse_id, tokens)
                if self._convert_bid_to_lang(bid) == 'fra':
                    lemmas = self.fra_lemmas_by_verse[verse_id]
                else:
                    lemmas = ['X'] * len(tokens)
                for token, lemma, sd in zip(tokens, lemmas, sds):
                    conllu_file.write(f'{long_verse_id}\t{token}\t{lemma}\t{sd}\n')
                conllu_file.write('\n')

    def _save_alignment_to_file(self):
        alignments_dir = '../external_repos/GNN-POSTAG/dataset/gdfa_final/gdfa/'
        # recursively convert defaultdict to dict
        self.pharaoh_alignments_by_bid_by_bid_by_verse_id = json.loads(
            json.dumps(self.pharaoh_alignments_by_bid_by_bid_by_verse_id))

        for verse_id in tqdm(self.pharaoh_alignments_by_bid_by_bid_by_verse_id,
                             desc=f'Saving alignments to {alignments_dir}',
                             total=len(self.pharaoh_alignments_by_bid_by_bid_by_verse_id)):
            alignments = self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id]
            file_path = os.path.join(alignments_dir, f'{verse_id}.txt.bin')
            pickle.dump(alignments, open(file_path, 'wb'))

    def remove_stop_words_from_alignment_file(self, stop_words_by_lang):
        alignments_dir = '../external_repos/GNN-POSTAG/dataset/gdfa_final/gdfa/'

        for verse_id, verse in tqdm(enumerate(self.verses_by_bid[self.source_bid]),
                                    desc=f'Updating alignments in {alignments_dir}',
                                    total=len(self.verses_by_bid[self.source_bid])):
            if verse.strip() == '':
                continue
            long_verse_id = str(self.verse_ids[verse_id])
            file_path = os.path.join(alignments_dir, f'{long_verse_id}.txt.bin')
            self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id] = pickle.load(open(file_path, 'rb'))
            for bid_1 in self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id]:
                for bid_2 in self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id][bid_1]:
                    alignment_line = self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id][bid_1][bid_2]
                    alignment_line = alignment_line.strip(' ')
                    aligned_wtxt_pairs = alignment_line.split(' ')

                    # convert to list of tuples of int
                    aligned_wtxt_pairs = [pair.split('-') for pair in aligned_wtxt_pairs]
                    aligned_wtxt_pairs = [(int(a), int(b)) for a, b in aligned_wtxt_pairs]

                    # remove word pair if one word is a stop word
                    lang_1 = self._convert_bid_to_lang(bid_1)
                    lang_2 = self._convert_bid_to_lang(bid_2)
                    verse_1 = self.wtxts_by_verse_by_bid[bid_1][verse_id]
                    verse_2 = self.wtxts_by_verse_by_bid[bid_2][verse_id]
                    aligned_wtxt_pairs = [pair for pair in aligned_wtxt_pairs
                                          if verse_1[pair[0]] not in stop_words_by_lang[lang_1]
                                          and verse_2[pair[1]] not in stop_words_by_lang[lang_2]]

                    # convert back to string
                    aligned_wtxt_pairs = [f'{pair[0]}-{pair[1]}' for pair in aligned_wtxt_pairs]
                    alignment_line = ' '.join(aligned_wtxt_pairs)

                    # update alignment
                    self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id][bid_1][bid_2] = alignment_line

            pickle.dump(self.pharaoh_alignments_by_bid_by_bid_by_verse_id[verse_id], open(file_path, 'wb'))
