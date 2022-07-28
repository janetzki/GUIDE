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

    def __init__(self, bibles_by_bid, score_threshold=0.5):
        self.bibles_by_bid = bibles_by_bid
        self.bids = list(bibles_by_bid.keys())
        self.source_bid = self.bids[0]
        self.source_lang = self._convert_bid_to_lang(self.source_bid)
        self.target_langs = sorted(set([self._convert_bid_to_lang(bid) for bid in self.bids]))
        self.all_langs = sorted(
            ['eng', 'fra', 'spa', 'ind', 'deu', 'rus', 'tha', 'tel', 'urd', 'hin', 'nep', 'vie', 'tpi', 'swp'])
        self.data_path = 'data'
        self.state_files_path = os.path.join(self.data_path, '0_state')
        self.tokenizer = 'bpe'
        self.vectorizer = TfidfVectorizer()
        self.state_loaded = False
        self.score_threshold = score_threshold

        # Saved data (preprocessing)
        self.sds_by_lang = {}
        self.verses_by_bid = {}
        self.words_by_text_by_lang = defaultdict(dict)
        self.question_by_qid_by_lang = defaultdict(dict)
        self.wtxts_by_verse_by_bid = {}

        # Saved data (mapping)
        self.aligned_wtxts_by_qid_by_lang_by_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

        # Not saved data (plotting)
        self.word_graph = None

        # Not saved data (predicting links)
        self.strength_by_lang_by_word = defaultdict(lambda: defaultdict(float))

        # Saved data (training)
        self.top_scores_by_qid_by_lang = defaultdict(dict)

        # Stores which variables have changed since they have last been saved to a file
        self.changed_variables = set()

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

    @staticmethod
    def _apply_prediction(graph, func, ebunch=None):
        """Applies the given function to each edge in the specified iterable
        of edges.
        """
        if ebunch is None:
            ebunch = nx.non_edges(graph)
        return ((u, v, func(u, v)) for u, v in ebunch)

    @staticmethod
    def _weighted_resource_allocation_index(graph, ebunch=None):
        r"""Compute the weighted resource allocation index of all node pairs in ebunch.

        References
        ----------
        .. resource allocati todo (mentioned on page 3 in https://www.nature.com/articles/srep12261.pdf) [1] T. Zhou, L. Lu, Y.-C. Zhang.
           Predicting missing links via local information.
           Eur. Phys. J. B 71 (2009) 623.
           https://arxiv.org/pdf/0901.0553.pdf
        """

        def predict(u, v):
            return sum((graph.get_edge_data(u, w)['weight'] + graph.get_edge_data(w, v)['weight'])
                       / graph.degree(w, weight='weight')
                       for w in nx.common_neighbors(graph, u, v))

        return DictionaryCreator._apply_prediction(graph, predict, ebunch)

    def _save_state(self):
        print('Saving state...')

        # save newly changed class variables to a separate dill file
        for variable_name in tqdm(self.changed_variables,
                                  desc='Saving class variables',
                                  total=len(self.changed_variables)):
            variable = getattr(self, variable_name)

            def save_file(key=''):
                if key:
                    file_path = os.path.join(self.state_files_path, f'{variable_name}_{key}.dill')
                else:
                    file_path = os.path.join(self.state_files_path, f'{variable_name}.dill')

                # make a backup copy
                # os.system(f'cp {file_path} {file_path}')

                with open(file_path, 'wb') as state_file:
                    if key:
                        dill.dump(variable[key], state_file)
                    else:
                        dill.dump(variable, state_file)

            # if type(variable) is dict or type(variable) is defaultdict:
            for key, value in tqdm(variable.items(),
                                   desc=f'Saving {variable_name}',
                                   total=len(variable),
                                   leave=True,
                                   position=0):
                save_file(key)
            # else:
            #     save_file()

        self.changed_variables.clear()
        print('State saved.')

    def _load_state(self):
        if self.state_loaded:
            return

        print('Loading state...')

        # load class variables from separate dill files
        for variable_name in ['sds_by_lang', 'verses_by_bid', 'words_by_text_by_lang', 'question_by_qid_by_lang',
                              'wtxts_by_verse_by_bid', 'aligned_wtxts_by_qid_by_lang_by_lang',
                              'top_scores_by_qid_by_lang']:
            variable = getattr(self, variable_name)
            if type(variable) is dict or type(variable) is defaultdict:
                # get all matching file names in directory
                file_names = os.listdir(os.path.join(self.state_files_path))
                file_paths = [os.path.join(self.state_files_path, file_name) for file_name in file_names if
                              file_name.startswith(variable_name)]

                for file_path in file_paths:
                    def load_file(fallback_value):
                        try:
                            return dill.load(state_file)
                        except EOFError:
                            print(f'{file_path} is broken. Skipping.')
                            return fallback_value

                    with open(file_path, 'rb') as state_file:
                        # if len(file_paths) > 1:
                        key = file_path.split('_')[-1].split('.')[0]
                        assert (key not in ('lang', 'bid'))
                        if key in self.target_langs or key in self.bids:
                            variable[key] = load_file(None)
                    # else:
                    #     variable = load_file(variable)

        self.top_scores_by_qid_by_lang = defaultdict(
            dict)  # activate this to switch between computing link scores and tf-idf scores
        self.state_loaded = True
        print('State loaded.')

    def _load_data(self):
        # load sds and bible verses for all languages
        languages = set([self._convert_bid_to_lang(bid) for bid in self.bids])
        for lang in tqdm(languages, desc='Loading semantic domains', total=len(languages)):
            # load sds
            if lang in self.sds_by_lang:
                print(f'Skipped: SDs for {lang} already loaded')
                continue

            sd_path = f'../semdom extractor/output/semdom_qa_clean_{lang}.csv'
            if os.path.isfile(sd_path):
                self.sds_by_lang[lang] = pd.read_csv(sd_path)
            else:
                print(f'WARNING: unable to load {sd_path}')
                # create empty dataframe
                self.sds_by_lang[lang] = pd.DataFrame(
                    {'cid': [], 'category': [], 'question_index': [], 'question': [], 'answer': []})
                assert (lang in ('deu', 'rus', 'vie', 'tpi'))
            self.changed_variables.add('sds_by_lang')

        for bid in tqdm(self.bids, desc='Loading bibles', total=len(self.bids)):
            # load bible verses
            if bid in self.verses_by_bid:
                print(f'Skipped: Bible {bid} already loaded')
                continue

            with open(os.path.join('../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid]),
                      'r') as bible:
                self.verses_by_bid[bid] = bible.readlines()
                self.changed_variables.add('verses_by_bid')
            assert (len(self.verses_by_bid[bid]) == 41899)

    def _build_sds(self):
        # convert sd dataframe to dictionary
        # optional: increase performance by querying wtxts from words_eng
        # optional: increase performance by using dataframe instead of dict
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
                    self.changed_variables.add('words_by_text_by_lang')

                self.question_by_qid_by_lang[lang][qid] = question
                self.changed_variables.add('question_by_qid_by_lang')

    def _tokenize_verses(self):
        for bid in tqdm(self.bids, desc='Tokenizing verses', total=len(self.bids)):
            if bid in self.wtxts_by_verse_by_bid:
                print(f'Skipped: Bible {bid} already tokenized')
                continue

            assert (self.tokenizer == 'bpe')
            file = os.path.join(
                '../load bibles in DGraph/content/scripture_public_domain', self.bibles_by_bid[bid])
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer()  # todo: fix tokenizer (e.g., splits 'Prahlerei' into 'Pra' and 'hlerei') (might not be so important because this mainly happens for rare words) possible solution: use pre-defined word list for English, using utoken instead did not significantly improve results
            tokenizer.train(files=[file], trainer=trainer)

            # tokenize all verses
            wtxts_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_bid[bid]]
            wtxts_by_verse = [tokenizer.encode(verse).tokens for verse in self.verses_by_bid[bid]]
            # lowercase all wtxts
            wtxts_by_verse = [[wtxt.lower() for wtxt in verse] for verse in wtxts_by_verse]

            self.wtxts_by_verse_by_bid[bid] = wtxts_by_verse.copy()
            self.changed_variables.add('wtxts_by_verse_by_bid')

            # mark words as appearing in the bible
            lang = self._convert_bid_to_lang(bid)
            wtxts_set = set([wtxt for wtxts in wtxts_by_verse for wtxt in wtxts])
            for wtxt in wtxts_set:
                if wtxt in self.words_by_text_by_lang[lang]:
                    self.words_by_text_by_lang[lang][wtxt].appears_in_bible = True
                else:
                    self.words_by_text_by_lang[lang][wtxt] = self.Word(wtxt, lang, [], True)
                self.changed_variables.add('words_by_text_by_lang')

    def _combine_alignments(self):
        # combine verses from two different bibles into a single file for wtxt aligner (fast_align)
        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2 and not (bid_1 == self.source_bid and bid_2 == self.source_bid):
                    # map every pair of different bibles plus the source bible to the source bible
                    continue

                aligned_bibles_file_path = f'{self.data_path}/diag_{bid_1}_{bid_2}_{self.tokenizer}.align'
                if os.path.isfile(aligned_bibles_file_path):
                    print(f'Skipped: Aligned bibles file {aligned_bibles_file_path} already exists')
                    continue

                combined_bibles_file_path = f'{self.data_path}/{bid_1}_{bid_2}_{self.tokenizer}.txt'
                with open(combined_bibles_file_path, 'w') as combined_bibles:
                    for idx, (bid_1_wtxts, bid_2_wtxts) in tqdm(enumerate(
                            zip(self.wtxts_by_verse_by_bid[bid_1], self.wtxts_by_verse_by_bid[bid_2])),
                            desc=f'Combining alignments for {bid_1} and {bid_2}',
                            total=len(self.wtxts_by_verse_by_bid[bid_1])):
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0 and len(bid_1_wtxts) + len(bid_2_wtxts) > 0:
                            # verse is missing in only one bible
                            print('Missing verse - verses might be misaligned!', idx, bid_1_wtxts, bid_2_wtxts)
                            pass
                        if len(bid_1_wtxts) * len(bid_2_wtxts) == 0:
                            # verse is missing in both bibles
                            bid_1_wtxts = ['#placeholder#']
                            bid_2_wtxts = ['#placeholder#']
                        combined_bibles.write(' '.join(bid_1_wtxts) + ' ||| ' + ' '.join(bid_2_wtxts) + '\n')

                print(subprocess.call(['sh', 'align_bibles.sh', bid_1, bid_2, self.tokenizer]))

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
        self.changed_variables.add('words_by_text_by_lang')

    def _map_word_to_qid(self, wtxt_1, wtxt_2, lang_1, lang_2, link_score=None, score_by_wtxt_by_qid_by_lang=None):
        for new_qid_1 in self.words_by_text_by_lang[lang_1][wtxt_1].qids:
            if link_score is None:
                self.aligned_wtxts_by_qid_by_lang_by_lang[lang_2][lang_1][new_qid_1] += ', ' + wtxt_2
                self.changed_variables.add('aligned_wtxts_by_qid_by_lang_by_lang')
            else:
                score_by_wtxt_by_qid_by_lang[lang_2][new_qid_1][wtxt_2] = link_score

    def _map_word_to_qid_bidirectionally(self, wtxt_1, wtxt_2, lang_1, lang_2, link_score=None,
                                         score_by_wtxt_by_qid_by_lang=None):
        self._map_word_to_qid(wtxt_1, wtxt_2, lang_1, lang_2, link_score, score_by_wtxt_by_qid_by_lang)
        if lang_1 == lang_2 and wtxt_1 == wtxt_2:
            # only map word once onto itself (loop edge)
            return
        self._map_word_to_qid(wtxt_2, wtxt_1, lang_2, lang_1, link_score, score_by_wtxt_by_qid_by_lang)

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
                self._map_word_to_qid_bidirectionally(wtxt_1, wtxt_2, lang_1, lang_2)
                self._add_directed_edge(wtxt_1, wtxt_2, lang_1, lang_2)
                self._add_directed_edge(wtxt_2, wtxt_1, lang_2, lang_1)

    def map_words_to_qids(self, load=False, save=False):
        # map words in all target language bibles to semantic domains
        if load:
            self._load_state()

        for bid_1 in self.bids:
            for bid_2 in self.bids:
                if bid_1 >= bid_2 and not (bid_1 == self.source_bid and bid_2 == self.source_bid):
                    # map every pair of different bibles plus the source bible to the source bible
                    continue
                with open(f'{self.data_path}/diag_{bid_1}_{bid_2}_{self.tokenizer}.align', 'r') as alignment_file:
                    alignment = alignment_file.readlines()
                    self._map_two_bibles(alignment, bid_1, bid_2)

        if save:
            self._save_state()

    def build_word_graph(self, load=False, save=False):
        if load:
            self._load_state()

        # flatmap words
        word_nodes = [word for lang in self.words_by_text_by_lang
                      for word in self.words_by_text_by_lang[lang].values()]

        # get all edges for alignments between words in flat list
        weighted_edges = []
        for lang_1 in self.words_by_text_by_lang:
            for word_1 in self.words_by_text_by_lang[lang_1].values():
                for word_2_str, count in word_1.aligned_words.items():
                    lang_2, wtxt_2 = word_2_str.split(': ')
                    if lang_2 not in self.target_langs:  # hacky
                        continue
                    word_2 = self.words_by_text_by_lang[lang_2][wtxt_2]
                    weighted_edges.append([word_1, word_2, count])

        # create graph structures with NetworkX
        self.word_graph = nx.Graph()
        self.word_graph.add_nodes_from(word_nodes)
        self.word_graph.add_weighted_edges_from(weighted_edges)

        if save:
            self._save_state()

    def plot_subgraph(self, lang, text, min_count=1):
        filtered_word_nodes = [word for word in self.word_graph.nodes if word.iso_language in self.target_langs]

        filtered_weighted_edges = []
        for edge in self.word_graph.edges(data='weight'):
            lang_1 = edge[0].iso_language
            lang_2 = edge[1].iso_language
            wtxt_1 = edge[0].text
            wtxt_2 = edge[1].text
            count = edge[2]
            if lang_1 not in self.target_langs or lang_2 not in self.target_langs \
                    or (lang_1 == lang_2 and wtxt_1 == wtxt_2) \
                    or (count < min_count and self._compute_link_score(edge[0], edge[1]) < self.score_threshold):
                continue
            filtered_weighted_edges.append(edge)

        filtered_word_graph = nx.Graph()
        filtered_word_graph.add_nodes_from(filtered_word_nodes)
        filtered_word_graph.add_weighted_edges_from(filtered_weighted_edges)

        # define filtered subgraph of neighbors of neighbors of node
        node = self.words_by_text_by_lang[lang][text]
        selected_nodes = {node}
        neighbors_1st_order = set()
        neighbors_2nd_order = set()
        neighbors_3rd_order = set()
        for neighbor_1st_order in filtered_word_graph.neighbors(node):
            neighbors_1st_order.add(neighbor_1st_order)
            for neighbor_2nd_order in filtered_word_graph.neighbors(neighbor_1st_order):
                if self._compute_link_score(neighbor_1st_order, neighbor_2nd_order) < self.score_threshold:
                    continue
                neighbors_2nd_order.add(neighbor_2nd_order)
                for neighbor_3rd_order in filtered_word_graph.neighbors(neighbor_2nd_order):
                    if self._compute_link_score(neighbor_2nd_order, neighbor_3rd_order) < self.score_threshold:
                        continue
                    neighbors_3rd_order.add(neighbor_3rd_order)
        # avoid that graph gets too large or messy for plotting
        max_nodes = 100
        if len(selected_nodes) + len(neighbors_1st_order) <= max_nodes:
            selected_nodes.update(neighbors_1st_order)
        if len(selected_nodes) + len(neighbors_2nd_order) <= max_nodes:
            selected_nodes.update(neighbors_2nd_order)
        if len(selected_nodes) + len(neighbors_3rd_order) <= max_nodes:
            selected_nodes.update(neighbors_3rd_order)
        G = filtered_word_graph.subgraph(selected_nodes)

        # set figure size heuristically
        width = len(selected_nodes) / 2.5
        plt.figure(figsize=(width, width))

        # use a different node color for each language
        palette = sns.color_palette('pastel')  # ('hls', len(self.target_langs))
        palette += [palette[2]] * 10  # hack to add color for 'vie' that is different from 'eng'
        palette = {lang: color for lang, color in zip(self.all_langs, palette)}
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
        for edge in G.edges(data='weight'):
            link_score = self._compute_link_score(edge[0], edge[1])
            color = 'green' if link_score >= self.score_threshold else 'gray'
            nx.draw_networkx_edges(G, pos=pos, edgelist=[edge],
                                   width=[math.log(edge[2]) + 1], alpha=0.5,
                                   edge_color=color)

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

    def _find_link_candidates(self):
        # # find all pairs of nodes in the word graph with a common neighbor
        # # (relevant if using the weighted resource allocation index)
        # link_candidates = set()
        # words_in_target_langs = [word for lang in self.target_langs
        #                             for word in self.words_by_text_by_lang[lang].values()]
        # for word_1 in tqdm(words_in_target_langs, desc='Finding link candidates', total=len(words_in_target_langs)):
        #     for common_neighbor in self.word_graph.neighbors(word_1):
        #         for word_2 in self.word_graph.neighbors(common_neighbor):
        #             if word_1 != word_2 \
        #                     and word_2.iso_language in self.target_langs \
        #                     and word_1.iso_language != word_2.iso_language \
        #                     and (word_2, word_1) not in link_candidates:
        #                 link_candidates.add((word_1, word_2))
        # return list(link_candidates)

        # find all neighboring nodes in the word graph
        link_candidates = set()
        words_in_target_langs = [word for lang in self.target_langs
                                 for word in self.words_by_text_by_lang[lang].values()]
        for word_1 in tqdm(words_in_target_langs, desc='Finding link candidates', total=len(words_in_target_langs)):
            for word_2 in self.word_graph.neighbors(word_1):
                if word_1 != word_2 \
                        and word_2.iso_language in self.target_langs \
                        and word_1.iso_language != word_2.iso_language \
                        and (word_2, word_1) not in link_candidates:
                    link_candidates.add((word_1, word_2))
        return list(link_candidates)

    def _compute_sum_of_weights(self, word_1, word_2):
        # compute sum of weights from word_1 to lang_2 words in neighbors of word_1
        lang_2 = word_2.iso_language

        # precompute strength sums to improve performance --> store them in cache self.strength_by_lang_by_word
        if lang_2 not in self.strength_by_lang_by_word[word_1]:
            self.strength_by_lang_by_word[word_1][lang_2] = sum(self.word_graph.get_edge_data(word_1, w)['weight']
                                                                for w in self.word_graph.neighbors(word_1)
                                                                if w.iso_language == lang_2)
        return self.strength_by_lang_by_word[word_1][lang_2]

    def _compute_link_score(self, word_1, word_2):
        # normalized edge weight = divide edge weight by the average sum of edge weights to words of the other language

        sum_weights_1_to_2 = self._compute_sum_of_weights(word_1, word_2)
        sum_weights_2_to_1 = self._compute_sum_of_weights(word_2, word_1)

        edge_weight = self.word_graph.get_edge_data(word_1, word_2)['weight']
        avg_weights_sum = (sum_weights_1_to_2 + sum_weights_2_to_1) / 2
        return edge_weight / avg_weights_sum

    def predict_links(self, load=False, save=False):
        if load:
            self._load_state()

        # link_candidates = [
        #     (self.words_by_text_by_lang['fra']['eau'],
        #      self.words_by_text_by_lang['deu']['wasser']),
        #     (self.words_by_text_by_lang['eng']['water'],
        #      self.words_by_text_by_lang['deu']['wasser']),
        #     (self.words_by_text_by_lang['eng']['water'],
        #      self.words_by_text_by_lang['fra']['eau']),
        #
        #     (self.words_by_text_by_lang['fra']['boire'],
        #      self.words_by_text_by_lang['deu']['trinken']),
        #     (self.words_by_text_by_lang['eng']['drink'],
        #      self.words_by_text_by_lang['deu']['trinken']),
        #     (self.words_by_text_by_lang['eng']['drink'],
        #      self.words_by_text_by_lang['fra']['boire']),
        # ]
        link_candidates = self._find_link_candidates()

        # preds = nx.jaccard_coefficient(self.word_graph)
        # preds = nx.adamic_adar_index(self.word_graph)
        # preds = self._weighted_resource_allocation_index(self.word_graph, link_candidates)
        # for u, v, link_score in tqdm(preds, desc='Predicting links', total=len(link_candidates)):

        score_by_wtxt_by_qid_by_lang = defaultdict(lambda: defaultdict(dict))
        for word_1, word_2 in tqdm(link_candidates, desc='Predicting links', total=len(link_candidates)):
            link_score = self._compute_link_score(word_1, word_2)
            self._map_word_to_qid_bidirectionally(word_1.text, word_2.text, word_1.iso_language, word_2.iso_language,
                                                  link_score, score_by_wtxt_by_qid_by_lang)

        self.top_scores_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_langs:
            if target_lang in self.top_scores_by_qid_by_lang:
                print(f'Skipped: top {target_lang} scores already collected')
                continue

            score_by_wtxt_by_qid = score_by_wtxt_by_qid_by_lang[target_lang]
            for qid, score_by_wtxt in tqdm(score_by_wtxt_by_qid.items(),
                                           desc=f'Collecting top {target_lang} scores',
                                           total=len(score_by_wtxt_by_qid)):
                score_by_wtxt = dict(sorted(score_by_wtxt.items(), key=lambda x: x[1], reverse=True))
                self.top_scores_by_qid_by_lang[target_lang][qid] = score_by_wtxt
                self.changed_variables.add('top_scores_by_qid_by_lang')

        if save:
            self._save_state()

    def train_tfidf_based_model(self, load=False, save=False):
        if load:
            self._load_state()

        for target_lang in self.target_langs:
            if target_lang in self.top_scores_by_qid_by_lang:
                print(f'Skipped: top {target_lang} tfidfs already collected')
                continue

            aligned_wtxts_by_qid = self.aligned_wtxts_by_qid_by_lang_by_lang[target_lang][self.source_lang]
            tfidfs = self.vectorizer.fit_transform(list(aligned_wtxts_by_qid.values()))
            assert (tfidfs.shape[0] == len(aligned_wtxts_by_qid))
            for idx, tfidf in tqdm(enumerate(tfidfs),
                                   desc=f'Collecting top {target_lang} tf-idf scores',
                                   total=tfidfs.shape[0]):  # caution: might fail in the debugger
                qid = list(aligned_wtxts_by_qid.keys())[idx]
                df = pd.DataFrame(tfidf.T.todense(), index=self.vectorizer.get_feature_names_out(), columns=['TF-IDF'])
                df = df.sort_values('TF-IDF', ascending=False)
                df = df[df['TF-IDF'] > 0]
                df = df.head(20)

                # convert df to dict
                scores_by_wtxt = {word: score for word, score in zip(df.index, df['TF-IDF'])}

                self.top_scores_by_qid_by_lang[target_lang][qid] = scores_by_wtxt
                self.changed_variables.add('top_scores_by_qid_by_lang')

        if save:
            self._save_state()

    def _filter_target_sds_with_threshold(self):
        # remove all target wtxts with a score (e.g., TF-IDF) below a threshold
        filtered_target_wtxts_by_qid_by_lang = defaultdict(dict)
        for target_lang in self.target_langs:
            for qid, score_by_wtxt in self.top_scores_by_qid_by_lang[target_lang].items():
                filtered_target_wtxts_by_qid_by_lang[target_lang][qid] = [key for key, value in score_by_wtxt.items() if
                                                                          value >= self.score_threshold]
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

        false_positives = []
        false_negatives = []
        for qid, wtxts in tqdm(predicted_target_wtxts_by_qid.items(),
                               desc=f'Counting true positive words in {target_lang} semantic domains',
                               total=len(predicted_target_wtxts_by_qid),
                               disable=True):
            num_positive_wtxts += len(wtxts)
            for wtxt in wtxts:
                num_true_positive_wtxts += wtxt in gt_target_wtxts_by_qid.get(qid, [])
                if wtxt not in gt_target_wtxts_by_qid.get(qid, []):
                    false_positives.append((wtxt, qid))
                    false_negatives.append((false_negative, qid) for false_negative in
                                           set(gt_target_wtxts_by_qid.get(qid, [])) - set(wtxts))

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
        # MRR improvement todo: Also filter out source wtxts (and questions, if empty) that do not appear in the source verses. (e.g., 'snake' does not appear in the KJV bible)
        # MRR improvement todo: also do this for source langs different from eng
        if target_lang == 'urd':
            return None
        target_qids = defaultdict(list)
        df_test = self._load_test_data(self.source_lang, target_lang)
        for source_wtxt in tqdm(self.words_by_text_by_lang[self.source_lang].values(),
                                desc=f'Filtering {target_lang} question ids',
                                total=len(self.words_by_text_by_lang[self.source_lang]),
                                disable=True):
            target_wtxts = list(df_test.query(f'source_wtxt=="{source_wtxt.text}"')['target_wtxts'])
            if len(target_wtxts) == 0:
                continue
            target_wtxts = target_wtxts[0]
            for qid in source_wtxt.qids:
                if qid in self.top_scores_by_qid_by_lang[target_lang]:
                    target_qids[qid].extend(target_wtxts)
                # some semantic domains are missing in the target sds because no aligned wtxts were found

        # in all selected target top_scores, look for first ranked target wtxt that also appears in df_test (gt data)
        mean_reciprocal_rank = 0
        for qid, target_wtxts in target_qids.items():
            wtxt_list = list(self.top_scores_by_qid_by_lang[target_lang][qid])
            reciprocal_rank = 0
            for idx, wtxt in enumerate(wtxt_list):
                if wtxt in target_wtxts:
                    reciprocal_rank = 1 / (idx + 1)
                    break
            if print_reciprocal_ranks:
                print(qid)
                print('\t', self.question_by_qid_by_lang[self.source_lang][qid])
                print('\t found (positive) words:', wtxt_list)
                print('\t reference (true) words:', target_wtxts)
                print('\t reciprocal rank:       ', f'{reciprocal_rank:.2f}\n')
            mean_reciprocal_rank += reciprocal_rank
        mean_reciprocal_rank /= len(target_qids)
        print(
            f'{len(target_qids)} / {len(self.top_scores_by_qid_by_lang[target_lang])} {target_lang} questions selected')
        print(f'MRR: {mean_reciprocal_rank:.3f}')

    def evaluate(self, load=False, print_reciprocal_ranks=False):
        if load:
            self._load_state()
        filtered_target_wtxts_by_qid_by_lang = self._filter_target_sds_with_threshold()
        print(f'\'=== Bibles: {self.bids}, Threshold: {self.score_threshold} ===')
        for target_lang in self.target_langs:
            print(f'\n\n--- Evaluation for {target_lang} ---')
            self._compute_f1_score(filtered_target_wtxts_by_qid_by_lang[target_lang], target_lang)
            self._compute_mean_reciprocal_rank(target_lang, print_reciprocal_ranks)


if __name__ == '__main__':
    dc = DictionaryCreator(bibles_by_bid={
        # 'bid-eng-asvbt': 'eng-engasvbt.txt',
        # 'bid-eng-asv': 'eng-eng-asv.txt',
        # 'bid-eng-BBE': 'eng-engBBE.txt',
        # 'bid-eng-Brenton': 'eng-eng-Brenton.txt',
        'bid-eng-DBY': 'eng-engDBY.txt',
        # 'bid-eng-DRA': 'eng-engDRA.txt',
        # 'bid-eng-gnv': 'eng-enggnv.txt',
        # 'bid-eng-jps': 'eng-engjps.txt',
        # 'bid-eng-kjv2006': 'eng-eng-kjv2006.txt',
        # 'bid-eng-kjvcpb': 'eng-engkjvcpb.txt',
        # 'bid-eng-kjv': 'eng-eng-kjv.txt',
        # 'bid-eng-lee': 'eng-englee.txt',
        # 'bid-eng-lxx2012': 'eng-eng-lxx2012.txt',
        # 'bid-eng-lxxup': 'eng-englxxup.txt',
        # 'bid-eng-noy': 'eng-engnoy.txt',
        # 'bid-eng-oebcw': 'eng-engoebcw.txt',
        # 'bid-eng-oebus': 'eng-engoebus.txt',
        # 'bid-eng-oke': 'eng-engoke.txt',
        # 'bid-eng-rv': 'eng-eng-rv.txt',
        # 'bid-eng-tnt': 'eng-engtnt.txt',
        # 'bid-eng-uk-lxx2012': 'eng-eng-uk-lxx2012.txt',
        # 'bid-eng-webbe': 'eng-eng-webbe.txt',
        # 'bid-eng-web-c': 'eng-eng-web-c.txt',
        # 'bid-eng-webpb': 'eng-engwebpb.txt',
        # 'bid-eng-webp': 'eng-engwebp.txt',
        # 'bid-eng-webster': 'eng-engwebster.txt',
        # 'bid-eng-web': 'eng-eng-web.txt',
        # 'bid-eng-wmbb': 'eng-engwmbb.txt',
        # 'bid-eng-wmb': 'eng-engwmb.txt',
        # 'bid-eng-Wycliffe': 'eng-engWycliffe.txt',
        # 'bid-eng-ylt': 'eng-engylt.txt',
        # 'bid-eng-niv11': 'extra_english_bibles/en-NIV11.txt',
        # 'bid-eng-niv84': 'extra_english_bibles/en-NIV84.txt',

        # 'bid-fra-fob': 'fra-fra_fob.txt',
        # 'bid-fra-lsg': 'fra-fraLSG.txt',

        # 'bid-spa': 'spa-spaRV1909.txt',
        # 'bid-ind': 'ind-ind.txt',
        # 'bid-tel': 'tel-telirv.txt',
        'bid-tha': 'tha-thaKJV.txt',
        # 'bid-hin': 'hin-hinirv.txt',
        # 'bid-nep': 'nep-nepulb.txt',
        # 'bid-urd': 'urd-urdgvu.txt',

        # 'bid-deu': 'no semdoms available/deu-deuelo.txt',
        # 'bid-rus': 'no semdoms available/rus-russyn.txt',,
        # 'bid-vie': 'no semdoms available/vie-vie1934.txt',
        # 'bid-tpi': 'no semdoms available/tpi-tpipng.txt',
        # 'bid-swp': 'no semdoms available/swp-swp.txt',
    }, score_threshold=0.2)

    load = False
    save = False
    dc.preprocess_data(load=load, save=save)
    dc.map_words_to_qids(load=load, save=save)

    # dc.build_word_graph(load=load, save=save)
    # dc.predict_links(load=load, save=save)
    # dc.plot_subgraph(lang='eng', text='water', min_count=4)

    dc.train_tfidf_based_model(load=load, save=save)
    dc.evaluate(load=load, print_reciprocal_ranks=False)
