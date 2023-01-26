import math
from collections import defaultdict

import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

from src.dictionary_creator.dictionary_creator import DictionaryCreator


class LinkPredictionDictionaryCreator(DictionaryCreator):
    STEPS = [
        '_preprocess_data',
        '_map_words_to_qids',
        '_build_word_graph (raw)',
        '_predict_lemmas',
        '_contract_lemmas',
        '_build_word_graph (contracted)',
        '_predict_translation_links',
        '_evaluate',
    ]

    LOADED_VARIABLES_BY_STEP = DictionaryCreator.LOADED_VARIABLES_BY_STEP | {
        '_build_word_graph (raw)': ['word_graph'],
        '_predict_lemmas': ['base_lemma_by_wtxt_by_lang', 'lemma_group_by_base_lemma_by_lang'],
        '_contract_lemmas': [],
        '_build_word_graph (contracted)': [],
        '_predict_translation_links': ['strength_by_lang_by_wtxt_by_lang', 'top_scores_by_qid_by_lang'],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.saved_variables.update({'word_graph', 'base_lemma_by_wtxt_by_lang', 'lemma_group_by_base_lemma_by_lang',
                                     'strength_by_lang_by_wtxt_by_lang', 'top_scores_by_qid_by_lang'})

        # Saved data (graph building)
        self.word_graph = None

        # Saved data (lemmas)
        self.base_lemma_by_wtxt_by_lang = defaultdict(dict)
        self.lemma_group_by_base_lemma_by_lang = defaultdict(lambda: defaultdict(set))

        # Saved data (predicting links)
        self.strength_by_lang_by_wtxt_by_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def _find_lemma_link_candidates(self):
        # find all pairs of nodes in the word graph with a common neighbor
        # (relevant if using the weighted resource allocation index)
        link_candidates = set()
        words_in_target_langs = [word for lang in self.target_langs
                                 for word in self.words_by_text_by_lang[lang].values()]
        for word_1 in tqdm(words_in_target_langs,
                           desc='Finding lemma link candidates',
                           total=len(words_in_target_langs)):
            for common_neighbor in self.word_graph.neighbors(word_1):
                for word_2 in self.word_graph.neighbors(common_neighbor):
                    if word_1 != word_2 \
                            and word_2.iso_language in self.target_langs \
                            and word_1.iso_language == word_2.iso_language \
                            and (word_2, word_1) not in link_candidates:
                        link_candidates.add((word_1, word_2))
        return list(link_candidates)

    def _find_translation_link_candidates(self):
        # find all neighboring nodes in the word graph
        link_candidates = set()
        words_in_target_langs = [word for lang in self.target_langs
                                 for word in self.words_by_text_by_lang[lang].values()]
        for word_1 in tqdm(words_in_target_langs,
                           desc='Finding translation link candidates',
                           total=len(words_in_target_langs)):
            for word_2 in self.word_graph.neighbors(word_1):
                if word_1 != word_2 \
                        and word_2.iso_language in self.target_langs \
                        and word_1.iso_language != word_2.iso_language \
                        and (word_2, word_1) not in link_candidates:
                    link_candidates.add((word_1, word_2))
        return list(link_candidates)

    def _compute_sum_of_weights(self, word_1, lang_2):
        # compute sum of weights from word_1 to target_lang words in neighbors of word_1
        # precompute strength sums to improve performance --> store them in cache self.strength_by_lang_by_wtxt_by_lang
        if lang_2 not in self.strength_by_lang_by_wtxt_by_lang[word_1.iso_language][word_1.text]:
            self.strength_by_lang_by_wtxt_by_lang[word_1.iso_language][word_1.text][lang_2] = sum(
                self.word_graph.get_edge_data(word_1, w)['weight']
                for w in self.word_graph.neighbors(word_1)
                if w.iso_language == lang_2)
        return self.strength_by_lang_by_wtxt_by_lang[word_1.iso_language][word_1.text][lang_2]

    def _compute_link_score(self, word_1, word_2):
        # normalized edge weight = divide edge weight by the average sum of edge weights to words of the other language

        sum_weights_1_to_2 = self._compute_sum_of_weights(word_1, word_2.iso_language)
        sum_weights_2_to_1 = self._compute_sum_of_weights(word_2, word_1.iso_language)

        edge_weight = self.word_graph.get_edge_data(word_1, word_2)['weight']
        avg_weights_sum = (sum_weights_1_to_2 + sum_weights_2_to_1) / 2
        return edge_weight / avg_weights_sum

    def _build_word_graph(self):
        # flatmap words
        word_nodes = [word for lang in self.words_by_text_by_lang
                      if lang in self.target_langs  # ignore additional languages in graph
                      for word in self.words_by_text_by_lang[lang].values()]

        # get all edges for alignments between words in flat list
        weighted_edges = set()
        for word_1 in word_nodes:
            for word_2, count in word_1.get_aligned_words_and_counts(self.words_by_text_by_lang):
                if word_2.iso_language not in self.target_langs:
                    continue
                weighted_edges.add((word_1, word_2, count))

        # create graph structures with NetworkX
        self.word_graph = nx.Graph()
        self.word_graph.add_nodes_from(word_nodes)
        self.word_graph.add_weighted_edges_from(weighted_edges)

    def _plot_subgraph(self, lang, text, min_count=1):
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

        # define filtered subgraph of a node's 1st, 2nd, and 3rd order neighbors
        node = self.words_by_text_by_lang[lang][text]
        selected_nodes = {node}
        neighbors_1st_order = set()
        neighbors_2nd_order = set()
        neighbors_3rd_order = set()
        for neighbor_1st_order in filtered_word_graph.neighbors(node):
            neighbors_1st_order.add(neighbor_1st_order)
            is_predicted_link_1st_order = self._compute_link_score(node, neighbor_1st_order) >= self.score_threshold
            for neighbor_2nd_order in filtered_word_graph.neighbors(neighbor_1st_order):
                if not is_predicted_link_1st_order \
                        and self._compute_link_score(neighbor_1st_order, neighbor_2nd_order) < self.score_threshold:
                    continue
                neighbors_2nd_order.add(neighbor_2nd_order)
                for neighbor_3rd_order in filtered_word_graph.neighbors(neighbor_2nd_order):
                    if self._compute_link_score(neighbor_2nd_order, neighbor_3rd_order) < self.score_threshold:
                        continue
                    neighbors_3rd_order.add(neighbor_3rd_order)

        # avoid that graph gets too large or messy for plotting
        max_nodes = 50
        selected_nodes.update(neighbors_1st_order)
        if len(selected_nodes) + len(neighbors_2nd_order) <= max_nodes:
            selected_nodes.update(neighbors_2nd_order)
            if len(selected_nodes) + len(neighbors_3rd_order) <= max_nodes:
                selected_nodes.update(neighbors_3rd_order)
        displayed_subgraph = filtered_word_graph.subgraph(selected_nodes)
        assert (len(displayed_subgraph.nodes) <= len(
            displayed_subgraph.edges) + 1)  # necessary condition if graph is connected

        # set figure size heuristically
        width = max(6, int(len(selected_nodes) / 2.2))
        plt.figure(figsize=(width, width))

        # use a different node color for each language
        palette = sns.color_palette('pastel')  # ('hls', len(self.target_langs))
        palette += [palette[2]] * 10  # hack to add color for 'vie' that is different from 'eng'
        palette = {lang: color for lang, color in zip(self.all_langs, palette)}
        node_colors = [palette[word.iso_language] for word in displayed_subgraph.nodes()]

        # show all the colors in a legend
        plt.legend(handles=[Patch(color=palette[lang], label=lang) for lang in self.target_langs])

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(displayed_subgraph)

        # draw nodes
        nx.draw_networkx_nodes(displayed_subgraph, pos=pos, node_color=node_colors)

        # draw only word texts as node labels
        nx.draw_networkx_labels(displayed_subgraph, pos=pos,
                                labels={word: self._transliterate_word(word) for word in displayed_subgraph.nodes()})

        # draw edges (thicker edges for more frequent alignments)
        for edge in displayed_subgraph.edges(data='weight'):
            link_score = self._compute_link_score(edge[0], edge[1])
            color = 'green' if link_score >= self.score_threshold else 'gray'
            nx.draw_networkx_edges(displayed_subgraph, pos=pos, edgelist=[edge],
                                   width=[math.log(edge[2]) + 1], alpha=0.5,
                                   edge_color=color)

        # draw edge labels
        edge_weights = nx.get_edge_attributes(displayed_subgraph, 'weight')
        if len(edge_weights):
            nx.draw_networkx_edge_labels(displayed_subgraph, pos, edge_labels=edge_weights)

        # plot the title
        title = 'Words that fast_align aligned with the\n' \
                f'{lang} word "{text}"'
        if min_count > 1:
            title += f' at least {min_count} times'
        plt.title(title)

        plt.show()

    def _weighted_resource_allocation_index(self, ebunch):
        r"""Compute the weighted resource allocation index of all node pairs in ebunch.

        References
        ----------
        .. resource allocation... todo
           (mentioned on page 3 in https://www.nature.com/articles/srep12261.pdf) [1] T. Zhou, L. Lu, Y.-C. Zhang.

           Predicting missing links via local information.
           Eur. Phys. J. B 71 (2009) 623.
           https://arxiv.org/pdf/0901.0553.pdf
        """

        def predict(word_1, word_2):
            return sum((self.word_graph.get_edge_data(word_1, common_neighbor)['weight'] +
                        self.word_graph.get_edge_data(common_neighbor, word_2)['weight'])
                       / (self._compute_sum_of_weights(common_neighbor, word_1.iso_language) +
                          self._compute_sum_of_weights(common_neighbor, word_2.iso_language))
                       for common_neighbor in nx.common_neighbors(self.word_graph, word_1, word_2)
                       if word_1.iso_language != common_neighbor.iso_language != word_2.iso_language)  # ignore
            # eng-eng-eng edges

        return DictionaryCreator._apply_prediction(predict, ebunch)

    def _predict_lemma_links(self, lemma_link_candidates):
        # preds = nx.jaccard_coefficient(self.word_graph)
        # preds = nx.adamic_adar_index(self.word_graph)
        preds = self._weighted_resource_allocation_index(lemma_link_candidates)
        for word_1, word_2, link_score in tqdm(preds, desc='Predicting lemma links', total=len(lemma_link_candidates)):
            assert word_1.iso_language == word_2.iso_language
            lang = word_1.iso_language
            wtxt_1 = word_1.text
            wtxt_2 = word_2.text

            if link_score < 0.01:
                continue
            distance = edit_distance(wtxt_1, wtxt_2)
            if distance < max(len(wtxt_1), len(wtxt_2)) / 3:
                # find the base lemma, which is the most frequent lemma
                base_lemma_1 = self.base_lemma_by_wtxt_by_lang[lang].get(wtxt_1, wtxt_1)
                base_lemma_2 = self.base_lemma_by_wtxt_by_lang[lang].get(wtxt_2, wtxt_2)
                words_by_text = self.words_by_text_by_lang[lang]

                # start with word_1 as the assumed base lemma
                new_base_lemma = wtxt_1
                if word_1.occurrences_in_bible < word_2.occurrences_in_bible:
                    # word_2 is more frequent than word_1
                    new_base_lemma = wtxt_2
                if words_by_text[new_base_lemma].occurrences_in_bible \
                        < words_by_text[base_lemma_1].occurrences_in_bible:
                    # word_1's base lemma is (even) more frequent
                    new_base_lemma = base_lemma_1
                if words_by_text[new_base_lemma].occurrences_in_bible \
                        < words_by_text[base_lemma_2].occurrences_in_bible:
                    # word_2's base lemma is (even) more frequent
                    new_base_lemma = base_lemma_2

                # build a group of all lemmas that belong together
                new_lemma_group = {wtxt_1, wtxt_2} \
                    .union(self.lemma_group_by_base_lemma_by_lang[lang][base_lemma_1]) \
                    .union(self.lemma_group_by_base_lemma_by_lang[lang][base_lemma_2])

                # update base lemmas and lemma groups
                # All lemmas point to the new base lemma. Only the base lemma points to the new lemma group.
                for wtxt in new_lemma_group:
                    self.base_lemma_by_wtxt_by_lang[lang][wtxt] = new_base_lemma
                    if wtxt in self.lemma_group_by_base_lemma_by_lang[lang]:
                        del self.lemma_group_by_base_lemma_by_lang[lang][wtxt]
                self.lemma_group_by_base_lemma_by_lang[lang][new_base_lemma] = new_lemma_group

        for lang in self.lemma_group_by_base_lemma_by_lang:
            # sort self.lemma_group_by_base_lemma_by_lang by key
            self.lemma_group_by_base_lemma_by_lang[lang] = defaultdict(
                set,
                sorted(self.lemma_group_by_base_lemma_by_lang[lang].items(), key=lambda x: x[0]))

    def _predict_lemmas(self):
        lemma_link_candidates = self._find_lemma_link_candidates()
        self._predict_lemma_links(lemma_link_candidates)

        # validate lemmas
        for lang in self.lemma_group_by_base_lemma_by_lang:
            for base_lemma, lemma_group in self.lemma_group_by_base_lemma_by_lang[lang].items():
                # check that all lemma groups contain at least the base lemma and another lemma
                assert base_lemma in lemma_group
                assert len(lemma_group) > 1

                # check that all lemmas point to the base lemma and that all non-base lemmas store no lemma group
                for lemma in lemma_group:
                    assert self.base_lemma_by_wtxt_by_lang[lang][lemma] == base_lemma
                    assert lemma == base_lemma or lemma not in self.lemma_group_by_base_lemma_by_lang[lang]

        for lang in self.target_langs:
            # if we found no lemmas for a language,
            # at least create an empty dictionary to show that we tried finding them
            if lang not in self.base_lemma_by_wtxt_by_lang:
                self.base_lemma_by_wtxt_by_lang[lang] = dict()
            if lang not in self.lemma_group_by_base_lemma_by_lang:
                self.lemma_group_by_base_lemma_by_lang[lang] = defaultdict(set)

    def _contract_lemmas(self):
        # merge lemmas in same lemma groups together into a single node
        assert (len(self.base_lemma_by_wtxt_by_lang) == len(self.target_langs) and
                len(self.lemma_group_by_base_lemma_by_lang) == len(self.target_langs))

        for lang in self.lemma_group_by_base_lemma_by_lang:
            if lang == 'eng':
                continue
                # todo?: We lemmatize English words using wordnet instead. --> todo: do not collect lemmas for English.

            for base_lemma_wtxt, lemma_wtxt_group in tqdm(self.lemma_group_by_base_lemma_by_lang[lang].items(),
                                                          desc=f'Contracting lemmas for {lang}',
                                                          total=len(self.lemma_group_by_base_lemma_by_lang[lang])):
                assert len(lemma_wtxt_group) > 1

                # collect words that belong to the same lemma group
                # by finding the corresponding words for each lemma text
                base_lemma_word = self.words_by_text_by_lang[lang][base_lemma_wtxt]
                lemma_group_words = set()
                for lemma_wtxt in lemma_wtxt_group:
                    if lemma_wtxt != base_lemma_wtxt:
                        lemma_group_words.add(self.words_by_text_by_lang[lang][lemma_wtxt])

                # contract words in the graph (i.e., merge all grouped lemma nodes into a single lemma group node)
                base_lemma_word.merge_words(lemma_group_words, self.words_by_text_by_lang,
                                            self.strength_by_lang_by_wtxt_by_lang)
                self.words_by_text_by_lang[lang][base_lemma_wtxt] = base_lemma_word

    def print_lemma_groups(self):  # pragma: no cover
        for lang in self.lemma_group_by_base_lemma_by_lang:
            for base_lemma, lemma_group in self.lemma_group_by_base_lemma_by_lang[lang].items():
                print(f'{lang} {base_lemma}: {lemma_group}')

    def _map_word_to_qids_with_score(self, source_wtxt, target_wtxt, source_lang, target_lang, link_score,
                                     score_by_wtxt_by_qid_by_lang):
        """
        Assign a target word to all of a source word's qids. A score indicates the assignment's strength.
        """
        for new_qid in self.words_by_text_by_lang[source_lang][source_wtxt].qids:
            score_by_wtxt = score_by_wtxt_by_qid_by_lang[target_lang][new_qid]
            if target_wtxt in score_by_wtxt:
                previous_score = score_by_wtxt[target_wtxt][0]
                if link_score > previous_score:
                    score_by_wtxt[target_wtxt] = (link_score, source_wtxt)
                # todo: find mathematically more elegant solution than using just the highest link score
                #  (something like 0.7 and 0.3 --> 0.9)
            else:
                score_by_wtxt[target_wtxt] = (link_score, source_wtxt)

    def _map_word_to_qid_bidirectionally_with_score(self, wtxt_1, wtxt_2, lang_1, lang_2, link_score,
                                                    score_by_wtxt_by_qid_by_lang):
        """
        Assign two words to all of each other's qids. A score indicates the assignment's strength.
        """
        assert lang_1 != lang_2
        self._map_word_to_qids_with_score(wtxt_1, wtxt_2, lang_1, lang_2, link_score, score_by_wtxt_by_qid_by_lang)
        self._map_word_to_qids_with_score(wtxt_2, wtxt_1, lang_2, lang_1, link_score, score_by_wtxt_by_qid_by_lang)

    def _predict_translation_links(self):
        link_candidates = self._find_translation_link_candidates()

        score_by_wtxt_by_qid_by_lang = defaultdict(lambda: defaultdict(dict))
        for word_1, word_2 in tqdm(link_candidates, desc='Predicting links', total=len(link_candidates)):
            link_score = self._compute_link_score(word_1,
                                                  word_2)  # todo: current: add early filtering with score threshold?
            self._map_word_to_qid_bidirectionally_with_score(word_1.text, word_2.text,
                                                             word_1.iso_language, word_2.iso_language,
                                                             link_score, score_by_wtxt_by_qid_by_lang)

        for target_lang in self.target_langs:

            # at least create an empty dictionary to show that we tried predicting links
            self.top_scores_by_qid_by_lang[target_lang] = dict()

            score_by_wtxt_by_qid = score_by_wtxt_by_qid_by_lang[target_lang]
            for qid, score_by_wtxt in tqdm(score_by_wtxt_by_qid.items(),
                                           desc=f'Collecting top {target_lang} scores',
                                           total=len(score_by_wtxt_by_qid)):
                score_by_wtxt = dict(sorted(score_by_wtxt.items(), key=lambda x: x[1][0], reverse=True))
                self.top_scores_by_qid_by_lang[target_lang][qid] = score_by_wtxt

    def create_dictionary(self, load=False, save=False, plot_word_lang='eng', plot_wtxt='drink', min_count=1,
                          print_reciprocal_ranks=False):
        self._execute_and_track_state(self._preprocess_data, load=load, save=save)
        self._execute_and_track_state(self._map_words_to_qids, load=load, save=save)

        # build the graph with single words as nodes
        self._execute_and_track_state(self._build_word_graph, step_name='_build_word_graph (raw)',
                                      load=load, save=save)

        self._plot_subgraph(lang=plot_word_lang, text=plot_wtxt, min_count=min_count)

        self._execute_and_track_state(self._predict_lemmas, load=load, save=save)
        self._execute_and_track_state(self._contract_lemmas, load=load, save=save)

        # build the word graph with lemma groups as nodes
        self._execute_and_track_state(self._build_word_graph, step_name='_build_word_graph (contracted)',
                                      load=load, save=save)

        self._execute_and_track_state(self._predict_translation_links, load=load, save=save)
        self._plot_subgraph(lang=plot_word_lang, text=plot_wtxt, min_count=min_count)
        self._execute_and_track_state(self._evaluate, print_reciprocal_ranks=print_reciprocal_ranks)
