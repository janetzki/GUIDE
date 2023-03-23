from collections import defaultdict

from src.dictionary_creator.tfidf_dictionary_creator import TfidfDictionaryCreator


class SemanticDomainIdentifier(object):
    def __init__(self, dc):
        self.dc = dc
        self.dc._load_state()

        self.lang = 'eng'

        # build a dictionary that maps wtxts to qids
        self.qid_by_wtxt = defaultdict(set)
        for qid in self.dc.top_scores_by_qid_by_lang[self.lang]:
            for wtxt, (score, source_wtxt) in self.dc.top_scores_by_qid_by_lang[self.lang][qid].items():
                if score < self.dc.score_threshold:
                    continue
                self.qid_by_wtxt[wtxt].add(qid)

    def identify_semantic_domains(self, verses):
        # lookup each token in the dictionary to identify semantic domains
        qids = list()
        for verse in verses:
            for wtxt in verse:
                if wtxt not in self.qid_by_wtxt:
                    continue
                for qid in self.qid_by_wtxt[wtxt]:
                    question = self.dc.question_by_qid_by_lang[self.lang][qid]
                    score, source_wtxt = self.dc.top_scores_by_qid_by_lang[self.lang][qid][wtxt]
                    qids.append((wtxt, qid, question, score))
        return qids

        # todo ~#1: evaluate identified semantic domains with verse-semdom mappings from human labeler


if __name__ == '__main__':  # pragma: no cover
    dc = TfidfDictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
    sdi = SemanticDomainIdentifier(dc)
    qids = sdi.identify_semantic_domains(sdi.dc.wtxts_by_verse_by_bid['bid-eng-DBY'])
