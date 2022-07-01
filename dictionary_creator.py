import math
import os
import subprocess
from collections import defaultdict, Counter

import dill
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from polyglot.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


class DictionaryCreator(object):
    class Word:
        def __init__(self, text, lang, qids=None, appears_in_bible=False):
            self.text = text
            self.iso_language = lang
            self.aligned_words = Counter()
            self.qids = set() if qids is None else qids
            self.appears_in_bible = appears_in_bible

        def __str__(self):
            return f'{self.iso_language}: {self.text}'

        def add_aligned_word(self, word):
            self.aligned_words[str(word)] += 1

    def __init__(self, bids):
        self.bids = bids
        self.target_langs = sorted(set([self._convert_bid_to_lang(bid) for bid in self.bids]))

        self.all_langs = sorted(['eng', 'fra', 'spa', 'ind', 'deu', 'rus', 'tha', 'tel', 'urd', 'hin', 'nep', 'vie'])
        self.bibles_by_bid = {
            'bid-eng-kjv': 'eng-eng-kjv.txt',
            'bid-eng-web': 'eng-eng-web.txt',
            'bid-fra-fob': 'fra-fra_fob.txt',
            'bid-fra-lsg': 'fra-fraLSG.txt',
            'bid-spa': 'spa-spaRV1909.txt',
            'bid-ind': 'ind-ind.txt',
            'bid-deu': 'no semdoms available/deu-deuelo.txt',
            'bid-rus': 'no semdoms available/rus-russyn.txt',
            'bid-tha': 'tha-thaKJV.txt',
            'bid-tel': 'tel-telirv.txt',
            'bid-urd': 'urd-urdgvu.txt',
            'bid-hin': 'hin-hinirv.txt',
            'bid-nep': 'nep-nepulb.txt',
            'bid-vie': 'no semdoms available/vie-vie1934.txt'
        }

        self.state_file_name = "dc_state.dill"  # f'dc_state-{self.bids}.dill'
        self.base_path = '../experiments'
        self.data_path = os.path.join(self.base_path, 'data')
        self.vectorizer = TfidfVectorizer()
        self.state_loaded = False

        # Saved data (preprocessing)
        self.sds_by_lang = None
        self.verses_by_bid = None
        self.words_by_text_by_lang = None
        self.question_by_qid_by_lang = None
        self.wtxts_by_verse_by_bid = None

        # Saved data (mapping)
        self.aligned_wtxts_by_qid_by_lang_by_lang = None

        # Saved data (plotting)
        self.word_graph = None

        # Saved data (training)
        self.top_tfidfs_by_qid_by_lang = None
        self.top_qids_by_wtxt_by_lang = None

    @staticmethod
    def _initialize_if_none(variable, default_value):
        # do not initialize variable if it already has been loaded from a file
        if variable is None:
            variable = default_value

    @staticmethod
    def _convert_bid_to_lang(bid):
        # example: bid-eng-kjv -> eng
        return bid[4:7]

    @staticmethod
    def _group_words_by_qid(words):
        wtxts_by_qid = defaultdict(list)
        for word in words.values():
            for qid in word.qids:
                wtxts_by_qid[qid].append(word.text)
        return wtxts_by_qid

    @staticmethod
    def _transliterate_word(word):
        if word.iso_language == 'hin':
            return ' '.join(Text(word.text, word.iso_language).transliterate('en'))
        return word.text

    def _save_state(self):
        # todo: save only changed subset of state to make this faster
        print('Saving state...')

        # make a backup copy of the dill file
        os.system(f'cp {self.data_path}/{self.state_file_name} {self.data_path}/{self.state_file_name}.bak')

        # save all class variables to a dill file
        with open(os.path.join(self.data_path, self.state_file_name), 'wb') as state_file:
            dill.dump({
                'sds_by_lang': self.sds_by_lang,
                'verses_by_bid': self.verses_by_bid,
                'words_by_text_by_lang': self.words_by_text_by_lang,
                'question_by_qid_by_lang': self.question_by_qid_by_lang,
                'wtxts_by_verse_by_bid': self.wtxts_by_verse_by_bid,
                'aligned_wtxts_by_qid_by_lang_by_lang': self.aligned_wtxts_by_qid_by_lang_by_lang,
                'word_graph': self.word_graph,
                'top_tfidfs_by_qid_by_lang': self.top_tfidfs_by_qid_by_lang,
                'top_qids_by_wtxt_by_lang': self.top_qids_by_wtxt_by_lang
            },
                state_file)
        print('State saved.')

    def _load_state(self):
        if self.state_loaded:
            return

        print('Loading state...')
        with open(os.path.join(self.data_path, self.state_file_name), 'rb') as state_file:
            state = dill.load(state_file)
            self.sds_by_lang = state['sds_by_lang']
            self.verses_by_bid = state['verses_by_bid']
            self.words_by_text_by_lang = state['words_by_text_by_lang']
            self.question_by_qid_by_lang = state['question_by_qid_by_lang']
            self.wtxts_by_verse_by_bid = state['wtxts_by_verse_by_bid']
            self.aligned_wtxts_by_qid_by_lang_by_lang = state['aligned_wtxts_by_qid_by_lang_by_lang']
            self.word_graph = state['word_graph']
            self.top_tfidfs_by_qid_by_lang = state['top_tfidfs_by_qid_by_lang']
            self.top_qids_by_wtxt_by_lang = state['top_qids_by_wtxt_by_lang']
        self.state_loaded = True
        print('State loaded.')

    def _load_data(self):
        # load sds and bible verses for all languages
        self._initialize_if_none(self.sds_by_lang, {})
        self._initialize_if_none(self.verses_by_bid, {})

        languages = set([self._convert_bid_to_lang(bid) for bid in self.bids])
        for lang in tqdm(languages, desc='Loading semantic domains', total=len(languages)):
            # load sds
            if lang in self.sds_by_lang:
                print(f'Skipped: SDs for {lang} already loaded')
                continue

            sd_path = f'{self.base_path}/../semdom extractor/output/semdom_qa_clean_{lang}.csv'
            if os.path.isfile(sd_path):
                self.sds_by_lang[lang] = pd.read_csv(sd_path)
            else:
                print(f'WARNING: unable to load {sd_path}')
                # create empty dataframe
                self.sds_by_lang[lang] = pd.DataFrame(
                    {'cid': [], 'category': [], 'question_index': [], 'question': [], 'answer': []})
                assert (lang in ('deu', 'rus', 'vie'))

        for bid in tqdm(self.bids, desc='Loading bibles', total=len(self.bids)):
            # load bible verses
            if bid in self.verses_by_bid:
                print(f'Skipped: Bible {bid} already loaded')
                continue

            with open(os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid]),
                      'r') as bible:
                self.verses_by_bid[bid] = bible.readlines()
            assert (len(self.verses_by_bid[bid]) == 41899)

    def _build_sds(self):
        # convert sd dataframe to dictionary
        # optional: increase performance by querying wtxts from words_eng
        # optional: increase performance by using dataframe instead of dict
        self._initialize_if_none(self.words_by_text_by_lang, defaultdict(dict))
        self._initialize_if_none(self.question_by_qid_by_lang, defaultdict(dict))

        for lang, sds in tqdm(self.sds_by_lang.items(), desc='Building semantic domains', total=len(self.sds_by_lang)):
            if lang in self.words_by_text_by_lang:
                print(f'Skipped: Words and SD questions for {lang} already loaded')
                continue

            for index, row in sds.iterrows():
                question = row.question.replace("'", '')
                question = question.replace('"', '')
                answer = row.answer.replace("'", '')
                answer = answer.replace('"', '')
                qid = f'{row.cid} {row.question_index}'
                words = {wtxt: self.Word(wtxt.strip(), lang, [qid]) for wtxt in answer.split(',') if wtxt}

                # add new words to words_by_text_by_lang
                for word in words.values():
                    if word.text not in self.words_by_text_by_lang[lang]:
                        self.words_by_text_by_lang[lang][word.text] = word
                    else:
                        self.words_by_text_by_lang[lang][word.text].qids.append(qid)

                self.question_by_qid_by_lang[lang][qid] = question

    def _tokenize_verses(self):
        self._initialize_if_none(self.wtxts_by_verse_by_bid, {})

        for bid in tqdm(self.bids, desc='Tokenizing verses', total=len(self.bids)):
            if bid in self.wtxts_by_verse_by_bid:
                print(f'Skipped: Bible {bid} already tokenized')
                continue

            file = os.path.join(
                '../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid])
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer()  # todo: fix tokenizer (e.g., splits 'Prahlerei' into 'Pra' and 'hlerei') (might not be so important because this mainly happens for rare words) possible solution: use pre-defined word list for English
            tokenizer.train(files=[file], trainer=trainer)

            # tokenize all verses
            wtxts_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_bid[bid]]
            # lowercase all wtxts
            wtxts_by_verse = [[wtxt.lower() for wtxt in verse] for verse in wtxts_by_verse]

            self.wtxts_by_verse_by_bid[bid] = wtxts_by_verse.copy()

            # mark words as appearing in the bible
            lang = self._convert_bid_to_lang(bid)
            wtxts_set = set([wtxt for wtxts in wtxts_by_verse for wtxt in wtxts])
            for wtxt in wtxts_set:
                if wtxt in self.words_by_text_by_lang[lang]:
                    self.words_by_text_by_lang[lang][wtxt].appears_in_bible = True
                else:
                    self.words_by_text_by_lang[lang][wtxt] = self.Word(wtxt, lang, [], True)

    def _combine_alignments(self):
        # combine verses from two different bibles into a single file for wtxt aligner
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2:
                    continue

                combined_bibles_file_path = f'{self.data_path}/{bid_1}-{bid_2}.txt'
                if os.path.isfile(combined_bibles_file_path):
                    print(f'Skipped: Combined bible {combined_bibles_file_path} already exists')
                    continue

                with open(combined_bibles_file_path, 'w') as combined_bibles:
                    for idx, (bid_1_wtxts, bid_2_wtxts) in tqdm(enumerate(
                            zip(self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2])),
                            desc=f'Combining alignments for {bid_1} and {bid_2}',
                            total=len(self.wtxts_by_verse_by_bid[bid_1])):
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0 and len(bid_1_wtxts) + len(bid_2_wtxts) > 0:
                            # print(idx)  # verse is missing in one bible
                            pass
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0:
                            bid_1_wtxts = ['#placeholder#']
                            bid_2_wtxts = ['#placeholder#']
                        combined_bibles.write(' '.join(bid_1_wtxts) + ' ||| ' + ' '.join(bid_2_wtxts) + '\n')

                if not os.path.isfile(f'{self.base_path}/data/diag_[{bid_1}]_[{bid_2}].align'):
                    print(subprocess.call(['sh', 'align_bibles.sh', bid_1, bid_2]))

    def preprocess_data(self, load=False, save=False):
        if load:
            self._load_state()
        self._load_data()
        self._build_sds()
        self._tokenize_verses()
        self._combine_alignments()
        if save:
            self._save_state()

    def _add_directed_edge(self, wtxt1, wtxt2, lang1, lang2):
        word1 = self.words_by_text_by_lang[lang1][wtxt1]
        word2 = self.words_by_text_by_lang[lang2][wtxt2]
        word1.add_aligned_word(word2)

    def _map_two_bibles(self, alignment, bid_1, bid_2):
        # map words in two bibles to semantic domains
        # Caveat: This function ignores wtxts that could not be aligned.
        lang_1 = self._convert_bid_to_lang(bid_1)
        lang_2 = self._convert_bid_to_lang(bid_2)

        if lang_2 in self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1]:
            print(f'Skipped: {bid_1} and {bid_2} already mapped')
            return

        for alignment_line, wtxts_1, wtxts_2 in tqdm(
                zip(alignment, self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2]),
                desc=f'Map {bid_1} and {bid_2} words and semantic domain questions bidirectionally',
                total=len(self.verses_by_bid[bid_1])):
            if alignment_line in ('\n', '0-0\n') and len(wtxts_1) * len(wtxts_2) == 0:
                continue
            aligned_wtxt_pairs = alignment_line.split(' ')
            aligned_wtxt_pairs[-1].replace('\n', '')

            for aligned_wtxt_pair in aligned_wtxt_pairs:
                wtxt_1_idx, wtxt_2_idx = [int(num) for num in aligned_wtxt_pair.split('-')]
                wtxt_1 = wtxts_1[wtxt_1_idx]
                wtxt_2 = wtxts_2[wtxt_2_idx]

                # add alignment edge and qids
                self._add_directed_edge(wtxt_1, wtxt_2, lang_1, lang_2)
                new_qids_1 = self.words_by_text_by_lang[lang_1][wtxt_1].qids
                for new_qid_1 in new_qids_1:
                    self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2][lang_1][new_qid_1] += ', ' + wtxt_2

                if lang_1 == lang_2 and wtxt_1 == wtxt_2:
                    continue

                # add alignment edge and qids for opposite direction
                self._add_directed_edge(wtxt_2, wtxt_1, lang_2, lang_1)
                new_qids_2 = self.words_by_text_by_lang[lang_2][wtxt_2].qids
                for new_qid_2 in new_qids_2:
                    self.aligned_wtxts_by_qid_by_lang_by_lang[lang_1][lang_2][new_qid_2] += ', ' + wtxt_1

    def map_target_words_to_qids(self, load=False, save=False):
        # map words in all target language bibles to semantic domains
        if load:
            self._load_state()

        self._initialize_if_none(self.aligned_wtxts_by_qid_by_lang_by_lang,
                                 defaultdict(lambda: defaultdict(lambda: defaultdict(str))))
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2:
                    continue
                with open(f'{self.data_path}/diag_[{bid_1}]_[{bid_2}].align', 'r') as alignment_file:
                    alignment = alignment_file.readlines()
                    self._map_two_bibles(alignment, bid_1, bid_2)

        if save:
            self._save_state()

    def plot_graph(self, load=False, lang='eng', text='tree', min_count=1):
        if load:
            self._load_state()

        # flatmap words
        word_nodes = [word for lang in self.words_by_text_by_lang
                      for word in self.words_by_text_by_lang[lang].values()]
        filtered_word_nodes = [word for word in word_nodes if word.iso_language in self.target_langs]

        # get all edges for alignments between words in flat list
        weighted_edges = []
        for lang_1 in self.words_by_text_by_lang:
            for word_1 in self.words_by_text_by_lang[lang_1].values():
                for word_2_str, count in word_1.aligned_words.items():
                    lang_2, wtxt_2 = word_2_str.split(': ')
                    word_2 = self.words_by_text_by_lang[lang_2][wtxt_2]
                    weighted_edges.append([word_1, word_2, count])
        filtered_weighted_edges = []
        for edge in weighted_edges:
            lang_1 = edge[0].iso_language
            lang_2 = edge[1].iso_language
            wtxt_1 = edge[0].text
            wtxt_2 = edge[1].text
            count = edge[2]
            if lang_1 not in self.target_langs or lang_2 not in self.target_langs \
                    or (lang_1 == lang_2 and wtxt_1 == wtxt_2) \
                    or count <= min_count:
                continue
            filtered_weighted_edges.append(edge)

        # create graph structures with NetworkX
        self.word_graph = nx.Graph()
        self.word_graph.add_nodes_from(word_nodes)
        self.word_graph.add_weighted_edges_from(weighted_edges)
        filtered_word_graph = nx.Graph()
        filtered_word_graph.add_nodes_from(filtered_word_nodes)
        filtered_word_graph.add_weighted_edges_from(filtered_weighted_edges)

        # set figure size
        plt.figure(figsize=(10, 10))

        # define subgraph
        node = self.words_by_text_by_lang[lang][text]
        neighbors = list(filtered_word_graph.neighbors(node)) + [node]
        G = filtered_word_graph.subgraph(neighbors)

        # use a different node color for each language
        palette = sns.color_palette('pastel')  # ('hls', len(self.target_langs))
        palette = {lang: color for lang, color in zip(self.all_langs, palette.extend(palette))}
        node_colors = [palette[word.iso_language] for word in G.nodes()]

        # show all the colors in a legend
        plt.legend(handles=[mpatches.Patch(color=palette[lang], label=lang) for lang in self.target_langs])

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(G)

        # draw nodes
        nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors)

        # draw only word texts as node labels
        nx.draw_networkx_labels(G, pos=pos, labels={word: self._transliterate_word(word) for word in G.nodes()})

        # draw edges (thicker edges for more frequent alignments
        widths = [math.log(e[2]) + 1 for e in G.edges(data='weight')]
        nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(), width=widths, alpha=0.5, edge_color='grey')

        # draw edge labels
        edge_weights = nx.get_edge_attributes(G, 'weight')
        if len(edge_weights):
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

        # plot the title
        title = f'Words that fast_align aligned with the {lang} word "{text}"'
        if min_count > 1:
            title += f' at least {min_count} times'
        plt.title(title)

        plt.show()

        return

    def predict_links(self, load=False):
        if load:
            self._load_state()

        preds = nx.jaccard_coefficient(self.word_graph)
        # preds = nx.adamic_adar_index(self.word_graph)
        # preds = nx.resource_allocation_index(self.word_graph)
        max_score = 0
        for u, v, p in preds:
            if u.iso_language in self.target_langs and v.iso_language in self.target_langs and p and p >= max_score:
                max_score = p
                print(f"({u}, {v}) -> {p:.3f}")

    def _build_top_tfidfs(self):
        self._initialize_if_none(self.top_tfidfs_by_qid_by_lang, defaultdict(dict))

        for target_lang in self.target_langs:
            if target_lang in self.top_tfidfs_by_qid_by_lang:
                print(f'Skipped: top {target_lang} tfidfs already collected')
                continue

            # merge alignments from all other aligned languages together
            merged_alignments = defaultdict(str)
            for lang in self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang]:
                if lang != 'eng':
                    continue
                for qid in self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][lang]:
                    merged_alignments[qid] += self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][lang][qid]

            tfidfs = self.vectorizer.fit_transform(list(merged_alignments.values()))
            assert (tfidfs.shape[0] == len(merged_alignments))
            for idx, tfidf in tqdm(enumerate(tfidfs), desc=f'Collecting top {target_lang} tf-idf scores',
                                   total=tfidfs.shape[0]):  # caution: might fail in the debugger
                qid = list(merged_alignments.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=['TF-IDF'])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                self.top_tfidfs_by_qid_by_lang[target_lang][qid] = df.head(20)

    def train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()
        self._build_top_tfidfs()

        # build self.top_qids_by_wtxt_by_lang
        self._initialize_if_none(self.top_qids_by_wtxt_by_lang, defaultdict(lambda: defaultdict(list)))
        for target_lang in self.target_langs:
            for qid, tfidfs_df in self.top_tfidfs_by_qid_by_lang[target_lang].items():
                for wtxt, tfidf in zip(list(tfidfs_df.index.values), list(tfidfs_df['TF-IDF'])):
                    self.top_qids_by_wtxt_by_lang[target_lang][wtxt].append((qid, tfidf))

        if save:
            self._save_state()

    def _filter_target_sds_with_threshold(self):
        # remove all target wtxts with a TF-IDF value below a threshold
        threshold = 0.15
        filtered_target_wtxts_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_langs:
            for qid, tfidfs_df in self.top_tfidfs_by_qid_by_lang[target_lang].items():
                filtered_target_wtxts_by_qid_by_lang[target_lang][qid] = list(
                    tfidfs_df[tfidfs_df['TF-IDF'] > threshold].index.values)
        return filtered_target_wtxts_by_qid_by_lang

    def _compute_f1_score(self, predicted_target_wtxts_by_qid, target_lang):
        """
        Compute precision, recall, and F1 score to evaluate DC. This requires a ground-truth semantic domain
        dictionary for the target language.
        """
        num_positive_wtxts = 0
        num_true_positive_wtxts = 0
        gt_target_wtxts_by_qid = self._group_words_by_qid(self.words_by_text_by_lang[target_lang])

        if len(gt_target_wtxts_by_qid) == 0:
            print(f'Cannot compute F1 score etc. for {target_lang} '
                  f'because no ground-truth target semantic domains have been loaded')
            return

        for qid, wtxts in tqdm(predicted_target_wtxts_by_qid.items(),
                               desc=f'Counting true positive words in {target_lang} semantic domains',
                               total=len(predicted_target_wtxts_by_qid),
                               disable=True):
            num_positive_wtxts += len(wtxts)
            for wtxt in wtxts:
                num_true_positive_wtxts += wtxt in gt_target_wtxts_by_qid.get(qid, [])

        # # How many non-unique wtxts are in the ground-truth target semantic domains?
        # num_total_sd_source_wtxts = 0
        # num_total_sd_wtxts_in_source_verses = 0
        # for _, wtxts_for_question in tqdm(self..items(),
        #                                   desc=f'Collecting words in {self.source_language} semantic domains',
        #                                   total=len(self.),
        #                                   disable=True):
        #     num_total_sd_source_wtxts += len(wtxts_for_question)
        #     overlap = wtxts_for_question & self.source_verse_wtxts_set
        #     num_total_sd_wtxts_in_source_verses += len(wtxts_for_question & self.source_verse_wtxts_set)

        # How many non-unique wtxts are in the ground-truth target semantic domains?
        num_total_gt_target_wtxts = 0
        num_total_gt_sd_wtxts_in_target_verses = 0
        # num_total_single_gt_sd_wtxts_in_target_verses = 0
        for wtxts_for_question in tqdm(gt_target_wtxts_by_qid.values(),
                                       desc=f'Collecting words in {target_lang} semantic domains',
                                       total=len(gt_target_wtxts_by_qid),
                                       disable=True):
            num_total_gt_target_wtxts += len(wtxts_for_question)
            overlap = [wtxt for wtxt in wtxts_for_question if
                       wtxt in self.words_by_text_by_lang[target_lang]
                       and self.words_by_text_by_lang[target_lang][wtxt].appears_in_bible]
            num_total_gt_sd_wtxts_in_target_verses += len(overlap)
            # single_wtxts = [wtxt for wtxt in overlap if ' ' not in wtxt]
            # num_total_single_gt_sd_wtxts_in_target_verses += len(single_wtxts)

        # How many of the found target wtxts actually appear in the ground-truth set?
        precision = num_true_positive_wtxts / num_positive_wtxts
        print(f'precision: {precision:.3f} ({num_true_positive_wtxts} '
              f'/ {num_positive_wtxts} found {target_lang} semantic domain words are correct)')

        # How many of the target sd wtxts in the ground-truth set were actually found?
        recall = num_true_positive_wtxts / num_total_gt_target_wtxts
        print(f'recall:    {recall:.3f} ({num_true_positive_wtxts} '
              f'/ {num_total_gt_target_wtxts} {target_lang} actual semantic domain words found)')

        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'F1:        {f1:.3f}')

        # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # was actually found?
        recall_adjusted = num_true_positive_wtxts / num_total_gt_sd_wtxts_in_target_verses
        print(f'recall*:   {recall_adjusted:.3f} ({num_true_positive_wtxts} '
              f'/ {num_total_gt_sd_wtxts_in_target_verses} {target_lang} actual semantic domain words '
              f'- that also appear in the target verses - found)')

        f1_adjusted = 2 * (precision * recall_adjusted) / (precision + recall_adjusted)
        print(f'F1*:       {f1_adjusted:.3f}\n')

        # How many of the gt target wtxts appear in the target verses?
        target_wtxt_coverage = num_total_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        print(
            f'Ground truth target word coverage: {target_wtxt_coverage:.3f} ({num_total_gt_sd_wtxts_in_target_verses} '
            f'/ {num_total_gt_target_wtxts} {target_lang} actual non-unique semantic domain words '
            f'also appear in the target verses)')

        # # How many of the source wtxts appear in the source verses?
        # source_wtxt_coverage = num_total_sd_wtxts_in_source_verses / num_total_sd_source_wtxts
        # print(f'Source wtxt coverage: {source_wtxt_coverage:.3f} ({num_total_sd_wtxts_in_source_verses} '
        #       f'/ {len(num_total_sd_source_wtxts)} {self.source_language} actual non-unique semantic domain words '
        #       'also appear in the source verses)')

        # optional: consider wtxt groups vs. single wtxts in calculation
        # # How many of the single gt target wtxts appear in the target verses?
        # target_wtxt_coverage = num_total_single_gt_sd_wtxts_in_target_verses / num_total_gt_target_wtxts
        # print(f'Ground truth single target wtxt coverage: {target_wtxt_coverage:.3f} ({num_total_single_gt_sd_wtxts_in_target_verses} '
        #       f'/ {num_total_gt_target_wtxts} {self.target_language} actual non-unique semantic domain words '
        #       'also appear in the target verses)')
        #
        # # How many of the target sd wtxts in the ground-truth set - that also appear in the target verses -
        # # was actually found?
        # recall_adjusted2 = num_true_positive_wtxts / num_total_single_gt_sd_wtxts_in_target_verses
        # print(f'recall**: {recall_adjusted2:.3f} ({num_true_positive_wtxts} '
        #       f'/ {num_total_single_gt_sd_wtxts_in_target_verses} {self.target_language} actual single semantic domain words '
        #       '- that also appear in the target verses - found)')
        #
        # f1_adjusted2 = 2 * (precision * recall_adjusted2) / (precision + recall_adjusted2)
        # print(f'F1**: {f1_adjusted2:.3f}')

    def _load_test_data(self, source_lang, target_lang):
        # load source and corresponding target wtxts from Purdue Team (ground truth data for dictionary creation)
        df_test = pd.read_csv(f'{self.data_path}/multilingual_semdom_dictionary.csv')
        df_test = df_test[[f'{source_lang}-000.txt', f'{target_lang}-000.txt']]
        df_test.columns = ['source_wtxt', 'target_wtxts']
        df_test = df_test[df_test['target_wtxts'].notna()]

        df_test['source_wtxt'] = df_test['source_wtxt'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test['target_wtxts'] = df_test['target_wtxts'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
        df_test = df_test.explode('source_wtxt').reset_index(drop=True)
        # df_test = df_test.groupby(['target_wtxts']).agg(list)
        return df_test

    def _compute_mean_reciprocal_rank(self, target_lang, print_reciprocal_ranks=False):
        # compute MRR to evaluate DC
        # Filter target question that we are going to check because ground truth set is limited:
        # We only consider questions which have at least one source wtxt in the gt set with a target translation.
        # todo: Also filter out source wtxts (and questions, if empty) that do not appear in the source verses. (e.g., 'snake' does not appear in the KJV bible)
        if target_lang == 'urd':
            return None
        target_qids = defaultdict(list)
        source_lang = 'eng'  # todo: also do this for other source_langs
        df_test = self._load_test_data(source_lang, target_lang)
        for source_wtxt in tqdm(self.words_by_text_by_lang[source_lang].values(),
                                desc=f'Filtering {target_lang} question ids',
                                total=len(self.words_by_text_by_lang[source_lang]),
                                disable=True):
            target_wtxts = list(df_test.query(f'source_wtxt=="{source_wtxt.text}"')['target_wtxts'])
            if len(target_wtxts) == 0:
                continue
            target_wtxts = target_wtxts[0]
            for qid in source_wtxt.qids:
                if qid in self.top_tfidfs_by_qid_by_lang[target_lang]:
                    target_qids[qid].extend(target_wtxts)
                # some semantic domains are missing in the target sds because no aligned wtxts were found

        # in all selected target top_tfidfs, look for first ranked target wtxt that also appears in df_test (gt data)
        mean_reciprocal_rank = 0
        for qid, target_wtxts in target_qids.items():
            wtxt_list = list(self.top_tfidfs_by_qid_by_lang[target_lang][qid].index)
            reciprocal_rank = 0
            for idx, wtxt in enumerate(wtxt_list):
                if wtxt in target_wtxts:
                    reciprocal_rank = 1 / (idx + 1)
                    break
            if print_reciprocal_ranks:
                print(qid)
                print('\t', self.question_by_qid_by_lang[source_lang][qid])
                print('\t found (positive) words:', wtxt_list)
                print('\t reference (true) words:', target_wtxts)
                print('\t reciprocal rank:       ', f'{reciprocal_rank:.2f}\n')
            mean_reciprocal_rank += reciprocal_rank
        mean_reciprocal_rank /= len(target_qids)
        print(
            f'{len(target_qids)} / {len(self.top_tfidfs_by_qid_by_lang[target_lang])} {target_lang} questions selected')
        print(f'MRR: {mean_reciprocal_rank:.3f}')

    def evaluate(self, load=False, print_reciprocal_ranks=False):
        if load:
            self._load_state()
        filtered_target_wtxts_by_qid_by_lang = self._filter_target_sds_with_threshold()
        for target_lang in self.target_langs:
            print(f'\n\n--- Evaluation for {target_lang} ---')
            self._compute_f1_score(filtered_target_wtxts_by_qid_by_lang[target_lang], target_lang)
            self._compute_mean_reciprocal_rank(target_lang, print_reciprocal_ranks)


if __name__ == '__main__':
    dc = DictionaryCreator([
        'bid-eng-kjv',
        # 'bid-eng-web',
        # 'bid-fra-fob',
        # 'bid-fra-lsg',
        # 'bid-spa',
        # 'bid-ind',
        # 'bid-tel',
        # 'bid-tha',
        # 'bid-hin',
        # 'bid-nep',
        # 'bid-urd',
        'bid-deu',
        # 'bid-rus',
        'bid-vie',
    ])
    dc.preprocess_data(load=True, save=True)
    dc.map_target_words_to_qids(load=True, save=True)
    dc.plot_graph(load=True, lang='deu', text='perle', min_count=1)
    # dc.predict_links(load=True)
    # dc.train_tfidf_based_model(load=True, save=True)
    # dc.evaluate(load=True, print_reciprocal_ranks=False)
