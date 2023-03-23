from collections import defaultdict

from src.dictionary_creator.tfidf_dictionary_creator import TfidfDictionaryCreator


class SemanticDomainIdentifier(object):
    @staticmethod
    def identify_semantic_domains():
        dc = TfidfDictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
        dc._load_state()
        lang = 'eng'

        # build a dictionary that maps wtxts to qids
        qid_by_wtxt = defaultdict(set)
        for qid in dc.top_scores_by_qid_by_lang[lang]:
            for wtxt, (score, source_wtxt) in dc.top_scores_by_qid_by_lang[lang][qid].items():
                if score < dc.score_threshold:
                    continue
                qid_by_wtxt[wtxt].add(qid)

        # lookup each token in the dictionary to identify semantic domains
        qids = list()
        for verse in dc.wtxts_by_verse_by_bid['bid-eng-DBY']:
            for wtxt in verse:
                if wtxt not in qid_by_wtxt:
                    continue
                for qid in qid_by_wtxt[wtxt]:
                    question = dc.question_by_qid_by_lang[lang][qid]
                    score, source_wtxt = dc.top_scores_by_qid_by_lang[lang][qid][wtxt]
                    qids.append((wtxt, qid, question, score))
        return qids

        # todo ~#1: evaluate identified semantic domains with verse-semdom mappings from human labeler


if __name__ == '__main__':
    sdi = SemanticDomainIdentifier()
    qids = sdi.identify_semantic_domains()
