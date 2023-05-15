from collections import defaultdict
import pandas as pd

from tqdm import tqdm

from src.dictionary_creator.tfidf_dictionary_creator import TfidfDictionaryCreator


class SemanticDomainIdentifier(object):
    def __init__(self, dc):
        self.dc = dc
        self.lang = 'eng'

        self.dc._load_state()

        with open('data/vref.txt', 'r') as f:
            self.vrefs = f.readlines()
        self.vrefs = [vref.strip() for vref in self.vrefs]

        # build a ground truth dictionary that maps wtxts to qids
        self.gt_qid_by_wtxt = defaultdict(set)
        for word in self.dc.words_by_text_by_lang[self.lang].values():
            for qid in word.qids:
                self.gt_qid_by_wtxt[word.text].add(qid)
        self.gt_qid_by_wtxt = dict(self.gt_qid_by_wtxt)

        # # build a dictionary that maps wtxts to qids
        # self.qid_by_wtxt = defaultdict(set)
        # for qid in self.dc.top_scores_by_qid_by_lang[self.lang]:
        #     for wtxt, (score, source_wtxt) in self.dc.top_scores_by_qid_by_lang[self.lang][qid].items():
        #         if score < self.dc.score_threshold:
        #             continue
        #         self.qid_by_wtxt[wtxt].add(qid)
        # self.qid_by_wtxt = dict(self.qid_by_wtxt)
        self.qid_by_wtxt = self.gt_qid_by_wtxt

        # remove all qids that start with a '9' (Grammar)
        for wtxt, qids in self.qid_by_wtxt.items():
            self.qid_by_wtxt[wtxt] = {qid for qid in qids if not qid.startswith('9')}

        self._tokenize_all_wtxts()

    def _build_direct_questions(self, question, tokens, verse, start_token_idx, token_count, verse_id):
        # format: "Does the word "TOKENS" in "token_1 token_2 TOKENS token_3 token 4" refer to a verb?"
        # e.g., "Does "ON THE SURFACE" in "darkness was ON THE SURFACE of the" indicate that something seems to be happening, or something seems to be true?"

        tokens = tokens.split('_')[0].upper()
        token_1 = verse[start_token_idx - 2] if 0 <= start_token_idx - 2 < len(verse) else ''
        token_2 = verse[start_token_idx - 1] if 0 <= start_token_idx - 1 < len(verse) else ''
        token_3 = verse[start_token_idx + token_count] if 0 <= start_token_idx + token_count < len(verse) else ''
        token_4 = verse[start_token_idx + token_count + 1] if 0 <= start_token_idx + token_count + 1 < len(
            verse) else ''

        # remove lemmatized tokens, keep originals
        token_1 = token_1.split('_')[0]
        token_2 = token_2.split('_')[0]
        token_3 = token_3.split('_')[0]
        token_4 = token_4.split('_')[0]

        # remove punctuation
        token_1 = token_1.strip('.,;:!?“”"')
        # token_2 = token_2.strip('.,;:!?“”"')
        # token_3 = token_3.strip('.,;:!?“”"')
        token_4 = token_4.strip('.,;:!?“”"')

        # remove tokens before and after punctuation
        token_1 = '' if token_2 in '.,;:!?“”"' else token_1
        token_4 = '' if token_3 in '.,;:!?“”"' else token_4

        context = f'{token_1} {token_2} {tokens} {token_3} {token_4}'.strip()
        bible_reference = self._convert_verse_id_to_bible_reference(verse_id)
        question = question.replace('# in ##', f'"{tokens}" in "{context}" ({bible_reference})')
        return question, context

    def _add_matched_qids(self, qids, matched_qids, verse, idx, token_count, tokens, answer, verse_id):
        for qid in qids:
            question_text = self.dc.question_by_qid_by_lang[self.lang][qid]
            cid, question_index = qid.split(' ')
            question = self.dc.sds_by_lang[self.lang].loc[
                self.dc.sds_by_lang[self.lang]['cid'] == cid].loc[
                self.dc.sds_by_lang[self.lang]['question_index'] == int(question_index)]
            assert len(question.values) == 1
            sd_name = question['category'].values[0]
            words = question['answer'].values[0]
            direct_question, context = self._build_direct_questions(question_text, tokens, verse, idx, token_count,
                                                                    verse_id)
            matched_qids[len(matched_qids)] = {'idx': idx, 'token_count': token_count, 'sd_name': sd_name,
                                               'question_text': question_text, 'words': words,
                                               'qid': qid, 'direct_question': direct_question,
                                               'tokens': tokens, 'context': context, 'answer': answer,
                                               'verse_id': verse_id}
            # score, source_wtxt = self.dc.top_scores_by_qid_by_lang[self.lang][qid][wtxt]
            # qids.append((wtxt, qid, question, score))

    @staticmethod
    def compute_statistics(matched_questions):
        # count qids, sort descending
        qid_count = defaultdict(int)
        for _, _, _, qid, _, question_text, _, _ in matched_questions:
            qid_count[(qid, question_text)] += 1
        qid_count = sorted(qid_count.items(), key=lambda x: x[1], reverse=True)
        top_qids_share = sum([count for _, count in qid_count[:250]]) / sum([count for _, count in qid_count])

        # count wtxts, sort descending
        wtxt_count = defaultdict(int)
        last_idx = None
        for idx, _, wtxt, _, _, _, _, _ in matched_questions:
            if idx != last_idx:
                wtxt_count[wtxt.split('_')[0]] += 1
                last_idx = idx
        wtxt_count = sorted(wtxt_count.items(), key=lambda x: x[1], reverse=True)
        top_wtxts_share = sum([count for _, count in wtxt_count[:250]]) / sum([count for _, count in wtxt_count])

        # count direct_questions, sort descending
        direct_question_count = defaultdict(int)
        for _, _, _, _, _, _, _, direct_question in matched_questions:
            direct_question_count[direct_question] += 1
        direct_question_count = sorted(direct_question_count.items(), key=lambda x: x[1], reverse=True)
        top_direct_questions_share = sum([count for _, count in direct_question_count[:250]]) / sum(
            [count for _, count in direct_question_count])

        return top_qids_share, top_wtxts_share, top_direct_questions_share, qid_count, wtxt_count, direct_question_count

    def identify_semantic_domains(self, verses, verse_id_start=0, verse_id_end=None):
        # lookup each token in the dictionary to identify semantic domains
        matched_qids = {}

        if verse_id_end is None:
            verse_id_end = len(verses)
        verse_subset = verses[verse_id_start:verse_id_end]
        for verse_offset, verse in enumerate(
                tqdm(verse_subset, desc='Identifying semantic domains', total=len(verse_subset))):
            verse_id = verse_id_start + verse_offset

            # move over the verse in a sliding window
            for window_length in range(5, 0, -1):
                for idx in range(len(verse) - window_length + 1):
                    qids = set()
                    # get the window
                    window = verse[idx:idx + window_length]
                    # get the window text
                    window_text = ' '.join([wtxt.split('_')[0] for wtxt in window])
                    window_text_lemmatized = ' '.join([wtxt.split('_')[-1] for wtxt in window])
                    # check if the window text is in the dictionary
                    if window_text in self.qid_by_wtxt:
                        qids |= self.qid_by_wtxt[window_text]
                    if window_text_lemmatized in self.qid_by_wtxt:
                        qids |= self.qid_by_wtxt[window_text_lemmatized]

                    # put trivial questions into a separate list
                    if len(qids) == 1:
                        # a question is trivial when it is the only question for these tokens (so we assume that the answer is 1 (= yes))
                        answer = '1'
                    else:
                        answer = ''
                    self._add_matched_qids(qids, matched_qids, verse, idx, window_length, window_text, answer,
                                           verse_id)

        return pd.DataFrame.from_dict(matched_qids, 'index')

    def _tokenize_all_wtxts(self):
        # add an extra space before and after each '’' in the keys of self.qid_by_wtxt to align tokenizers
        self.qid_by_wtxt = {wtxt.replace('’', ' ’ '): qids for wtxt, qids in self.qid_by_wtxt.items()}


    def _convert_verse_id_to_bible_reference(self, verse_id):
        # e.g., 0 -> Gen 1:1, 23213 -> Mat 1:1
        vref = self.vrefs[verse_id]
        vref = vref[0] + vref[1].lower() + vref[2].lower() + vref[3:]
        return vref

    def convert_bible_reference_to_verse_id(self, book, chapter, verse):
        # e.g., GEN 1:1 -> 0, MAT 1:1 -> 23213
        vref = f'{book} {chapter}:{verse}'
        return self.vrefs.index(vref)


def update_matched_questions(question_by_qid):
    # updates phrasing of direct_question matched_questions.csv
    matched_questions = pd.read_csv('data/2_sd_labeling/matched_questions.csv')

    for idx, row in tqdm(matched_questions.iterrows(), desc='Updating direct questions', total=len(matched_questions)):
        qid = row['qid']
        tokens = row['tokens']
        context = row['context']
        bible_reference = 'N/A' # verse_id_to_bible_reference(row['verse_id'])
        question = question_by_qid[qid]
        direct_question = question.replace('# in ##', f'"{tokens.upper()}" in "{context} ({bible_reference})"')
        matched_questions.loc[idx, 'direct_question'] = direct_question

    matched_questions.to_csv('data/2_sd_labeling/matched_questions.csv', index=False)


if __name__ == '__main__':  # pragma: no cover
    dc = TfidfDictionaryCreator(['bid-eng-web', 'bid-deu'], score_threshold=0.2)
    sdi = SemanticDomainIdentifier(dc)

    # update_matched_questions(sdi.dc.question_by_qid_by_lang['eng'])

    verses = sdi.dc.wtxts_by_verse_by_bid['bid-eng-web']
    matched_questions = sdi.identify_semantic_domains(verses, 23213, 26992)  # all four gospels # 0, 31170 = skip apocrypha

    # select relevant columns
    matched_questions = matched_questions[['direct_question', 'qid', 'tokens', 'context', 'answer', 'verse_id']]

    # # group questions, and count the number of verse_ids
    # matched_questions = matched_questions.groupby(['direct_question', 'qid', 'tokens', 'context', 'answer']).agg(
    #     {'verse_id': 'count'}).reset_index()
    #
    # # sort by count
    # matched_questions = matched_questions.sort_values(by='verse_id', ascending=False)

    matched_questions.to_csv('data/2_sd_labeling/matched_questions_gospels_raw.csv', index=False)

    print(sdi.compute_statistics(matched_questions))
