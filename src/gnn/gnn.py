import gc
import math
import os
import pprint
import re
import textwrap
from collections import defaultdict, deque
from functools import partial

import _pickle as cPickle
import editdistance as editdistance
import imageio as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import wandb
from matplotlib import colors
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from src.dictionary_creator.link_prediction_dictionary_creator import LinkPredictionDictionaryCreator


# Although there are many LRL lemmas (e.g., "entends", "entendra"), they are not a big problem since its unlikely that a false negative edge between a lemma and a sd gets into the training data.
# Todo: use edge types, eg. with RGCNConv / RGATConv?

# NOTES
# Der Unterschied zwischen edge_index und edge_label_index ist, dass edge_index die Kanten für den Encoder sind, während
# edge_label_index die Kanten für den Decoder sind.
# Der Encoder erstellt ein Node Embedding für jeden Knoten mit den Kanten aus edge_index. Daher darf er keine negativen Kanten bekommen.
# Der Decoder wird dann auf binärer Klassifikation von Kanten trainiert. Daher braucht er positive und negative Kanten.
# "edge_label_index will be used for the decoder to make predictions
# and edge_label will be used for model evaluation."
# edge_index contains no negative edges.
# Der Encoder sieht alle Kanten auf dem Plot. todo?: fra-sd Kanten entfernen, die erst der val decoder und test decoder sehen dürfen.
# Der Decoder sieht nur die farbigen Kanten.

def compute_f1_score_with_threshold(target, predicted, threshold):
    predicted = predicted >= threshold
    return f1_score(target, predicted)


def compute_precision_and_recall_with_threshold(target, predicted, threshold):
    predicted = predicted >= threshold
    return precision_score(target, predicted), recall_score(target, predicted)


class F1_Loss(torch.nn.Module):
    # Adapted version of https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''Calculate F1 score. Can work with gpu tensors

    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, unknown_ids):
        # assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.sigmoid(y_pred)

        tp = (y_true * y_pred)
        # tn = ((1 - y_true) * (1 - y_pred))
        fp = ((1 - y_true) * y_pred)  # this consumes a lot of memory
        fn = (y_true * (1 - y_pred))

        # do not count positives and negatives at unknown ids
        # use unknown_ids as two-dimensional mask
        tp[unknown_ids] = 0
        fp[unknown_ids] = 0
        fn[unknown_ids] = 0

        tp = tp.sum().to(torch.float32)
        fp = fp.sum().to(torch.float32)
        fn = fn.sum().to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1


class SoftPrecision(torch.nn.Module):
    # Only precision, does not make sense as loss function
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, unknown_ids):
        y_pred = F.sigmoid(y_pred)

        tp = (y_true * y_pred)
        fp = ((1 - y_true) * y_pred)

        # do not count positives and negatives at unknown ids
        # use unknown_ids as two-dimensional mask
        tp[unknown_ids] = 0
        fp[unknown_ids] = 0

        tp = tp.sum().to(torch.float32)
        fp = fp.sum().to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        return precision


class SoftRecall(torch.nn.Module):
    # Only recall, does not make sense as loss function
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, unknown_ids):
        y_pred = F.sigmoid(y_pred)

        tp = (y_true * y_pred)
        fn = (y_true * (1 - y_pred))

        # do not count positives and negatives at unknown ids
        # use unknown_ids as two-dimensional mask
        tp[unknown_ids] = 0
        fn[unknown_ids] = 0

        tp = tp.sum().to(torch.float32)
        fn = fn.sum().to(torch.float32)

        recall = tp / (tp + fn + self.epsilon)
        return recall


f1_loss_function = F1_Loss().cuda()
soft_precision_function = SoftPrecision().cuda()
soft_recall_function = SoftRecall().cuda()


def get_neg_edges_for_sd(word_idxs, forbidden_edges, sd_idx):
    assert type(forbidden_edges) is set
    neg_edges = list()
    for word_idx in word_idxs:
        if (sd_idx, word_idx) not in forbidden_edges \
                and (sd_idx, word_idx) not in forbidden_edges:
            neg_edges.append((sd_idx, word_idx))
    return neg_edges


class Model(torch.nn.Module):
    # https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.plot_file_paths = []

        self.conv1 = GCNConv(in_channels, out_channels, add_self_loops=False, normalize=False, improved=True, bias=bias)

        # Smart initialize weight matrix
        offset = in_channels - out_channels
        with torch.no_grad():
            for x in range(out_channels):
                # Set diagonal (identity matrix for the question feature)
                self.conv1.lin.weight[x][x + offset] = 50.0  # 2.0
                # # Set (weighted) node degree feature
                # self.conv1.lin.weight[x][0] = 0.01
                # self.conv1.lin.weight[x][1] = 0.01

        # Initialize bias vector
        with torch.no_grad():
            torch.nn.init.constant_(self.conv1.bias, -5.0)  # -3.0)

        # self.visualize_weights('initial')

    def forward(self, x, edge_index, edge_weight):
        z = self.conv1(x, edge_index, edge_weight)
        return z

    def visualize_weights(self, title, epoch=None, loss=None, plot_path=None):
        print('Visualizing weights...')

        # set size to 2000 x 2000 pixels
        plt.rcParams["figure.figsize"] = (20, 20)

        weight_matrix = self.conv1.lin.weight.detach().cpu()

        # Convert the matrix tensors to a NumPy array
        weight_matrix_np = weight_matrix.numpy()

        # Create heatmaps using Matplotlib
        norm = colors.TwoSlopeNorm(vmin=-5.0, vcenter=0, vmax=5.0)
        plt.imshow(weight_matrix_np, cmap='bwr', norm=norm, interpolation='nearest')
        plt.colorbar()
        if epoch is not None and loss is not None:
            title += f' (epoch {epoch}, loss {loss:.3f})'
        plt.title('Weight Matrix Heatmap ' + title)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        if epoch is not None:
            file_name = f'plot_weight_matrix_{epoch}.png'
            file_path = os.path.join(plot_path, file_name)
            self.plot_file_paths.append(file_path)
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

            if self.conv1.bias is not None:
                bias_matrix = self.conv1.bias.detach().cpu()
                bias_matrix_np = bias_matrix.numpy()

                # Add 1 dimension to the vectors
                bias_matrix_np = np.expand_dims(bias_matrix_np, axis=1)

                # bias has only one dimension, so we have to make it a 2D array to plot it
                bias_matrix_np = np.repeat(bias_matrix_np, repeats=100, axis=0)
                plt.imshow(bias_matrix_np, cmap='bwr', norm=norm, interpolation='nearest')
                plt.colorbar()
                plt.title('Bias Vector Heatmap ' + title)
                plt.xlabel('Columns')
                plt.ylabel('Rows')
                plt.show()

        # reset size to 400 x 400 pixels
        plt.rcParams["figure.figsize"] = (4, 4)
        print('Done visualizing weights.')

    def print_highest_qid_correlations(self, qid_node_names, num_max=20):
        print('Collecting highest correlated qids...')
        weight_matrix = self.conv1.lin.weight.detach().cpu()
        # create a priority queue for num_max elements with the highest weights
        correlated_qids = deque(maxlen=num_max)
        correlated_qids_in_id_matrix = deque(maxlen=num_max)
        offset = weight_matrix.size()[1] - weight_matrix.size()[0]
        assert weight_matrix.size()[0] == len(qid_node_names)
        for out_channel in range(len(qid_node_names)):
            for in_channel in range(len(qid_node_names)):
                weight = weight_matrix[out_channel][in_channel + offset].item()
                in_qid = qid_node_names[in_channel]
                out_qid = qid_node_names[out_channel]

                # add the weight to the queue if it is higher than the lowest weight in the queue
                if out_channel == in_channel:
                    if len(correlated_qids_in_id_matrix) < num_max or weight > \
                            correlated_qids_in_id_matrix[0][0]:
                        bias = self.conv1.bias[out_channel].item() if self.conv1.bias is not None else 0.0
                        correlated_qids_in_id_matrix.append((weight, in_qid, out_qid, bias))
                    continue

                # add the weight to the queue if it is higher than the lowest weight in the queue
                if len(correlated_qids) < num_max or weight > correlated_qids[0][0]:
                    correlated_qids.append((weight, in_qid, out_qid))

        # sort the correlated qids
        correlated_qids = sorted(correlated_qids, key=lambda x: x[0], reverse=True)
        correlated_qids_in_id_matrix = sorted(correlated_qids_in_id_matrix, key=lambda x: x[0], reverse=True)
        print('Highest correlated qids:')
        pprint.pprint(correlated_qids, sort_dicts=False)

        print('\nHighest correlated qids in identity matrix:')
        pprint.pprint(correlated_qids_in_id_matrix, sort_dicts=False)


class EarlyStopper:
    # https://stackoverflow.com/a/73704579/8816968
    def __init__(self, patience=1, min_delta=0.0, warm_up=30):
        self.patience = patience
        self.min_delta = min_delta
        self.warm_up = warm_up
        self.counter = 0
        self.min_validation_loss = np.inf
        self.last_checkpoint = None

    def early_stop(self, validation_loss, epoch, save_model_function, checkpoint_path):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            save_model_function(checkpoint_path)
            self.last_checkpoint = checkpoint_path
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and epoch > self.warm_up:
                return True
        return False


class LinkPredictor(object):
    def __init__(self, dc, target_langs, config, graph_path, ignored_langs=None):
        self.dc = dc
        self.wandb_original_run_name = wandb.run.name
        self.target_langs = target_langs
        self.all_langs = dc.target_langs
        self.ignored_langs = {} if ignored_langs is None else ignored_langs
        self.gt_langs = self.target_langs

        self.node_labels_training = list(self.gt_langs)
        self.node_labels_training.append('semantic_domain_question')
        self.node_labels_all = list(self.all_langs)
        self.node_labels_all.append('semantic_domain_question')

        self.graph_path = graph_path
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.optimizer = None
        self.palette = None
        self.train_loaders = None
        self.df_additional_words = None

        # saved graph
        self.G = None
        self.empty_questions_by_lang = None
        self.num_removed_nodes = 0

        # saved dataset
        self.data = None
        self.sd_node_idx_by_name = None
        self.word_node_idx_by_name = None
        self.sd_node_names = None
        self.word_node_names = None
        self.word_node_idxs_by_lang = None
        self.word_node_names_by_sd_name = None
        self.forbidden_neg_train_edges = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.color_masks = None

    def build_network(self):
        # load graph from file if it exists
        if os.path.exists(self.graph_path):
            print(f'Loading graph from {self.graph_path}')
            with open(self.graph_path, 'rb') as f:
                gc.disable()  # disable garbage collection to speed up loading
                self.G, self.empty_questions_by_lang, self.num_removed_nodes = cPickle.load(f)
                gc.enable()
            print('Done loading graph')
            return self.G

        self.G = nx.Graph()
        self.dc._load_state()

        long_qid_name_by_qid = {}  # e.g., '1.1.1.1 1' -> 'sd: 1.1.1.1 1 Moon'
        empty_questions = {
            '1.5.4 3': 'Does # in ## refer to a species of moss?',
            '1.5.4 5': 'Does # in ## refer to a species of liverworts are there?',
            '3.4.1.4.5 8': "Does # in ## refer to what people say when they don't care?",
            '3.5.1.7.1 5': 'Does # in ## refer to what people say to thank someone?',
            '3.5.1.7.1 6': 'Does # in ## refer to what people say when someone thanks them?',
            '4.9.5.3 1': 'Does # in ## refer to relating to God or to the spirits?',
            '5.7.1 6': 'Does # in ## refer to what you say when you are beginning to go to sleep?',
            '5.7.1 10': 'Does # in ## refer to what someone says to someone else who is going to sleep?',
            '6.6.4.4 1': 'Does # in ## refer to working?',
            '8.4.5.1 10': 'Does # in ## refer to a noun phrase with no overt marker?',
        }
        qids = list(self.dc.question_by_qid_by_lang['eng'].keys())
        qids.extend(list(empty_questions.keys()))
        qids = [qid for qid in qids if not qid.startswith('9')]  # remove grammar SDs

        for qid in tqdm(qids, total=len(qids)):
            cid = qid.split(' ')[0]
            question_idx = int(qid.split(' ')[1])
            sd_name = self.dc.sds_by_lang['eng'][self.dc.sds_by_lang['eng']['cid'] == cid]['category'].iloc[0]
            try:
                question = self.dc.sds_by_lang['eng'][(self.dc.sds_by_lang['eng']['cid'] == cid) & (
                        self.dc.sds_by_lang['eng']['question_index'] == question_idx)]['question'].iloc[0]
            except IndexError:
                question = empty_questions[qid]
            long_qid_name = f'qid: {sd_name} {qid} ({question})'  # f'sd: {sd_name}'
            long_qid_name_by_qid[qid] = long_qid_name

        # add nodes for qids
        for qid in qids:
            self.G.add_node(long_qid_name_by_qid[qid], lang='semantic_domain_question')

        # add nodes for words and edges to semantic domains and alignments
        nodes_by_lang = defaultdict(set)
        edges = set()
        graph_langs = set(self.all_langs) - set(self.ignored_langs)
        for lang in tqdm(graph_langs,
                         desc='Building network',
                         total=len(graph_langs)):
            for word in self.dc.words_by_text_by_lang[lang].values():
                # add word-qid edges
                for qid in word.qids:
                    if qid.startswith('9'):
                        continue
                    long_qid_name = long_qid_name_by_qid[qid]
                    assert long_qid_name in self.G.nodes
                    edges.add((str(word), long_qid_name, 1))

                nodes_by_lang[lang].add(str(word))

                # add word-word edges
                for translation, alignment_count in word.get_aligned_words_and_counts(
                        self.dc.words_by_text_by_lang):
                    translation_lang = translation.iso_language
                    if translation_lang == lang or translation_lang in self.ignored_langs:
                        # skip self-loops and ignored langs
                        continue
                    nodes_by_lang[translation_lang].add(str(translation))

                    # # skip words that only belong to qids that are not in the graph
                    # if not any(long_qid_name_by_qid[qid] in self.G.nodes for qid in translation.qids):
                    #     continue

                    # skip rare edges for higher precision and speed
                    if alignment_count < self.config['min_alignment_count']:
                        continue

                    edges.add((str(word), str(translation), alignment_count))

        # add eng words first
        self.G.add_nodes_from(nodes_by_lang['eng'], lang='eng')
        for lang in self.gt_langs:
            # add nodes for gt langs first
            self.G.add_nodes_from(nodes_by_lang[lang], lang=lang)

        for lang in nodes_by_lang.keys() - self.gt_langs:
            # add remaining nodes
            self.G.add_nodes_from(nodes_by_lang[lang], lang=lang)
        self.G.add_weighted_edges_from(edges)

        # assert that self.G has no node in self.ignore_langs
        for node in self.G.nodes:
            assert self.G.nodes[node]['lang'] not in self.ignored_langs

        self.normalize_edge_weights()
        self.filter_edges_by_weight()
        self.find_empty_questions_by_lang([n for n in long_qid_name_by_qid.values() if n in self.G])

        # remove target language nodes with no edge to a GT lang node because we cannot predict their semantic domain
        for lang in self.all_langs:
            for node in nodes_by_lang[lang]:
                if len([n for n in self.G.neighbors(node) if self.G.nodes[n]['lang'] in self.gt_langs]) == 0:
                    self.G.remove_node(node)
                    self.num_removed_nodes += 1

        print(f'Saving graph to {self.graph_path}')
        with open(self.graph_path, 'wb') as f:
            cPickle.dump((self.G, self.empty_questions_by_lang, self.num_removed_nodes), f)
        print('Done saving graph')

    def normalize_edge_weights(self):
        for edge in tqdm(self.G.edges(data='weight'), desc='Normalizing edge weights', total=len(self.G.edges)):
            u, v = edge[0], edge[1]
            lang1 = self.G.nodes[u]['lang']
            lang2 = self.G.nodes[v]['lang']
            if 'semantic_domain_question' in (lang1, lang2):
                continue
            word1 = self.dc.words_by_text_by_lang[lang1][u.split(': ')[1]]
            word2 = self.dc.words_by_text_by_lang[lang2][v.split(': ')[1]]
            self.G[u][v]['weight'] = self.dc._compute_link_score(word1, word2)

    def filter_edges_by_weight(self):
        # filter out all edges between words with a small weight
        # (This does not compromise the recall metric.)
        removed_edges = []
        for edge in tqdm(self.G.edges(data='weight'), desc='Filtering edges by weight', total=len(self.G.edges)):
            if edge[2] < self.config['min_edge_weight']:
                removed_edges.append(edge)
        for edge in removed_edges:
            self.G.remove_edge(edge[0], edge[1])

    def find_empty_questions_by_lang(self, long_qid_names):
        # for each lang, find all qids that have no edge to a word in that lang
        self.empty_questions_by_lang = dict()
        for lang in self.all_langs:
            self.empty_questions_by_lang[lang] = set()
        for lang in tqdm(self.gt_langs, desc='Finding empty questions by lang', total=len(self.gt_langs)):
            if lang not in self.gt_langs:
                self.empty_questions_by_lang[lang] = long_qid_names
                continue
            for long_qid_name in long_qid_names:
                neighbors = [n for n in self.G.neighbors(long_qid_name) if self.G.nodes[n]['lang'] == lang]
                if len(neighbors) == 0:
                    self.empty_questions_by_lang[lang].add(long_qid_name)

    def convert_empty_questions_by_lang_to_tensors(self):
        for lang in self.all_langs:
            temp = torch.zeros(len(self.sd_node_idx_by_name), dtype=torch.bool)
            for long_qid_name in self.empty_questions_by_lang[lang]:
                qid_idx = self.sd_node_idx_by_name[long_qid_name]
                temp[qid_idx] = True
            self.empty_questions_by_lang[lang] = temp

    def plot_subgraph(self, graph, node):
        non_latin_languages = {'nep', 'hin', 'mal'}  # difficult to plot
        target_langs = set(self.all_langs) - non_latin_languages
        target_langs.add('semantic_domain_question')
        self.palette = sns.color_palette('pastel')
        self.palette += self.palette
        self.palette = {lang: color for lang, color in zip(target_langs, self.palette)}
        filtered_word_nodes = [word for word in graph.nodes() if graph.nodes[word]['lang'] in target_langs]

        filtered_weighted_edges = []
        for edge in graph.edges(data='weight'):
            lang_1 = graph.nodes[edge[0]]['lang']
            lang_2 = graph.nodes[edge[1]]['lang']
            # wtxt_1 = edge[0][5:]
            # wtxt_2 = edge[1][5:]
            # count = edge[2]
            if lang_1 not in target_langs or lang_2 not in target_langs:
                # or (lang_1 == lang_2 and wtxt_1 == wtxt_2)
                # or (count < min_count and self._compute_link_score(edge[0], edge[1]) < score_threshold):
                continue
            filtered_weighted_edges.append(edge)

        filtered_word_graph = nx.Graph()
        filtered_word_graph.add_nodes_from(filtered_word_nodes)
        filtered_word_graph.add_weighted_edges_from(filtered_weighted_edges)

        # also add the data of the original graph to the filtered graph
        for n in filtered_word_graph.nodes():
            filtered_word_graph.nodes[n].update(graph.nodes[n])
        for e in filtered_word_graph.edges():
            filtered_word_graph.edges[e].update(graph.edges[e])

        # define filtered subgraph of a node's 1st, 2nd, and 3rd order neighbors
        selected_nodes = {node}
        neighbors_1st_order = set()
        neighbors_2nd_order = set()
        neighbors_3rd_order = set()
        for neighbor_1st_order in filtered_word_graph.neighbors(node):
            neighbors_1st_order.add(neighbor_1st_order)
            # for neighbor_2nd_order in filtered_word_graph.neighbors(neighbor_1st_order):
            #     neighbors_2nd_order.add(neighbor_2nd_order)
            #     for neighbor_3rd_order in filtered_word_graph.neighbors(neighbor_2nd_order):
            #         neighbors_3rd_order.add(neighbor_3rd_order)

        # avoid that graph gets too large or messy for plotting
        max_nodes = 50
        selected_nodes.update(neighbors_1st_order)
        if len(selected_nodes) + len(neighbors_2nd_order) <= max_nodes:
            selected_nodes.update(neighbors_2nd_order)
            if len(selected_nodes) + len(neighbors_3rd_order) <= max_nodes:
                selected_nodes.update(neighbors_3rd_order)
            else:
                # only add semantic domain nodes
                selected_nodes.update(
                    [n for n in neighbors_3rd_order if graph.nodes[n]['lang'] == 'semantic_domain_question'])
        else:
            # only add semantic domain nodes
            selected_nodes.update(
                [n for n in neighbors_2nd_order if graph.nodes[n]['lang'] == 'semantic_domain_question'])
        displayed_subgraph = filtered_word_graph.subgraph(selected_nodes)
        assert (len(displayed_subgraph.nodes) <= len(
            displayed_subgraph.edges) + 1)  # necessary condition if graph is connected

        # set figure size heuristically
        width = max(6, int(len(selected_nodes) * 1.5))  # / 2.2))
        plt.figure(figsize=(width, width))

        # use a different node color for each language
        node_colors = [self.palette[data['lang']] for node_name, data in displayed_subgraph.nodes(data=True)]

        # show all the colors in a legend
        plt.legend(handles=[Patch(color=self.palette[lang], label=lang) for lang in target_langs])

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(displayed_subgraph)

        # draw nodes
        nx.draw_networkx_nodes(displayed_subgraph, pos=pos, node_color=node_colors)

        # draw only word texts as node labels
        nx.draw_networkx_labels(displayed_subgraph, pos=pos,
                                labels={node_name: '\n'.join(textwrap.wrap(node_name.split(': ')[1], width=20))
                                        for node_name in displayed_subgraph.nodes()})

        # draw edges (thicker edges for more frequent alignments)
        for edge in displayed_subgraph.edges(data='weight'):
            weight = edge[2]

            if 'semantic_domain_question' == graph.nodes[edge[0]]['lang']:
                color, _, _, _ = self.get_edge_color(edge[1], edge[0])
            elif 'semantic_domain_question' == graph.nodes[edge[1]]['lang']:
                color, _, _, _ = self.get_edge_color(edge[0], edge[1])
            else:
                color = 'black'
            if color == 'red':
                continue

            nx.draw_networkx_edges(displayed_subgraph, pos=pos, edgelist=[edge],
                                   # caution: might fail in debug mode with Python 3.10 instead of Python 3.9 or 3.11
                                   width=[math.log(weight * 10) + 1], edge_color=color, alpha=0.5)

        # draw edge labels with 2 decimal places
        edge_weights = nx.get_edge_attributes(displayed_subgraph, 'weight')
        edge_labels = dict([((u, v), f"{d['weight']:.2f}") for u, v, d in displayed_subgraph.edges(data=True)])
        # skip edges to semantic domains
        edge_labels = {k: v for k, v in edge_labels.items() if
                       graph.nodes[k[0]]['lang'] != 'semantic_domain_question' and
                       graph.nodes[k[1]]['lang'] != 'semantic_domain_question'}
        if len(edge_weights):
            nx.draw_networkx_edge_labels(displayed_subgraph, pos, edge_labels=edge_labels)

        plt.title(f'Nodes close to "{node}"')
        plt.show()

    def train(self):
        ys = []
        y_hats = []
        # node_idxs = []
        train_losses = []
        self.model.train()
        for train_loader in self.train_loaders:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch.to(self.device)
                x = batch.x.clone()
                edge_index = batch.edge_index
                edge_weight = batch.edge_weight
                y = batch.y[:batch.batch_size]
                # x = x.to_dense()
                x[:batch.batch_size] = 0  # mask out the nodes to be predicted
                y_hat = self.model(x, edge_index, edge_weight)[:batch.batch_size]
                train_loss = self.compute_loss_with_empty_questions(y_hat, y, batch.n_id[:batch.batch_size])
                ys.append(y.detach().cpu())
                y_hats.append(y_hat.detach().cpu())
                # node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())
                train_losses.append(train_loss.detach().cpu())
                train_loss.backward()
                self.optimizer.step()

        y = torch.cat(ys, dim=0)
        y_hat = torch.cat(y_hats, dim=0)
        # node_idxs = torch.cat(node_idxs, dim=0)
        assert (len(y) == len(y_hat))
        assert (len(y) == self.data.train_mask.sum())
        train_loss = torch.stack(train_losses).mean().item()
        return train_loss

    def train_link_predictor(self):
        early_stopper = EarlyStopper(patience=self.config['patience'], warm_up=self.config['warm_up'])
        for epoch in range(1, self.config['epochs'] + 1):
            plots = ['weights'] if epoch % 10 == 0 else []
            val_loss, _, _, _, precision, recall, _, f1, f1_loss, val_soft_precision, val_soft_recall = self.evaluate(
                self.data.val_mask, threshold=0.99, compute_ideal_threshold=False, plots=plots, num_frame=epoch)

            train_loss = self.train()
            wandb.log({"train loss": train_loss, "val loss": val_loss,
                       "val soft precision": val_soft_precision, "val soft recall": val_soft_recall})
            wandb.watch(self.model)

            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Soft Precision: {val_soft_precision:.3f}, Val Soft Recall: {val_soft_recall:.3f}")

            checkpoint_path = f"model_{wandb.run.name}_epoch{epoch}_val-loss{val_loss:.3f}_checkpoint.bin"
            if early_stopper.early_stop(val_loss, epoch, self.save_model, checkpoint_path):
                print("Early stopping")
                self.create_gif()
                self.load_model(early_stopper.last_checkpoint)
                break

    def compute_loss_with_empty_questions(self, y_hat, y, node_idxs, function=None):
        # do not count candidates at unknown positions
        node_langs = [self.word_node_names[i.item()].split(': ')[0] for i in node_idxs]
        empty_questions = [self.empty_questions_by_lang[lang] for lang in node_langs]
        empty_questions = torch.stack(empty_questions).to(self.device)
        if function is None:
            return self.criterion(y_hat, y, empty_questions)
        else:
            return function(y_hat, y, empty_questions)

    def compute_metric_with_empty_questions(self, candidates, node_idxs):
        # do not count candidates at unknown positions
        node_langs = [self.word_node_names[i.item()].split(': ')[0] for i in node_idxs]
        empty_questions = [self.empty_questions_by_lang[lang] for lang in node_langs]
        valid_questions = torch.logical_not(torch.stack(empty_questions)).to(self.device)
        candidates = candidates.to(self.device)
        result_tensor = torch.logical_and(candidates, valid_questions)
        result_num = torch.sum(result_tensor).item()
        return result_num, result_tensor

    def get_node_names_and_label_from_edge_label_index(self, edge_label, edge_label_index, idx):
        label = edge_label[idx].item()
        sd_idx = edge_label_index[(0, idx)].item()
        word_idx = edge_label_index[(1, idx)].item()
        sd = self.sd_node_names[sd_idx]
        word = self.word_node_names[word_idx]
        return sd, word, word_idx, label

    def find_similar_words_in_sd(self, word, sd_name):
        # use word_node_names_by_sd_name and normalized edit distance to find similar words
        # return a list of similar words
        similar_words = []
        word_plain = word.split(': ')[1]
        for w in self.word_node_names_by_sd_name[sd_name]:
            w_plain = w.split(': ')[1]
            dist = editdistance.eval(word_plain, w_plain) / len(word_plain)
            if dist <= 0.4:
                similar_words.append((w, dist))
        similar_words.sort(key=lambda x: x[1])
        return similar_words

    def print_human_readable_predictions(self, y, y_hat, link_probs, mask, threshold, max_print_words=10,
                                         save_to_csv_path=None, word_idx_map=None, max_print_total=100, ):
        print("Printing human readable predictions...")
        predictions = []
        # for each element in out, lookup all edges in eval_data[self.TARGET_EDGE_TYPE].edge_label_index
        count_printed_words = 0
        count_printed_total = 0
        assert (mask is None or len(mask) == len(link_probs))
        for word_idx, qid_preds in tqdm(enumerate(link_probs), total=len(link_probs)):
            if mask is not None and not mask[word_idx]:
                continue
            printed = False
            for qid_idx, prob in enumerate(qid_preds):
                pred = 1 if prob >= threshold else 0
                if pred == 0:  # and is_correct:
                    continue
                label = y[word_idx][qid_idx] if y is not None else None
                is_correct = label is None or (pred == label).item()

                original_word_idx = word_idx_map[word_idx].item() if word_idx_map is not None else word_idx
                word = self.word_node_names[original_word_idx]
                sd = self.sd_node_names[qid_idx]
                pred_raw = y_hat[word_idx][qid_idx]
                assert not is_correct or word in self.word_node_names_by_sd_name[sd]

                prefix = '! ' if not is_correct and label is not None else '  '  # highlight false positives and false negatives
                similar_words = []
                explanation = ''
                if pred == 1 and not is_correct:
                    similar_words = self.find_similar_words_in_sd(word, sd)
                    explanation = self.explain_false_positive(original_word_idx, qid_idx)
                if not save_to_csv_path:
                    print(
                        f'{prefix} prediction: {pred} ({pred_raw:.0f} --> {prob:.2f}/{threshold:.2f}), actual: {label}, {word} ({word_idx}) <---> {sd} ({qid_idx})',
                        similar_words, explanation)
                printed = True
                count_printed_total += 1
                if pred == 1:
                    predictions.append([sd, word, prob.item(), pred_raw.item(), is_correct])
            count_printed_words += printed
            if count_printed_words >= max_print_words or count_printed_total >= max_print_total:
                print(f'... and {len(link_probs) - max_print_words} more words')
                break
        print(f'Printed {count_printed_words} words and {count_printed_total} total predictions')

        if save_to_csv_path is None:
            return

        print(f'Saving predictions to {save_to_csv_path}...')
        with open(save_to_csv_path, 'w') as f:
            f.write(
                'key,qid,semantic domain,language,word,question,score,raw score,is correct,similar words\n')
            for prediction in predictions:
                sd_qid_question = prediction[0].split(': ')[1]
                semantic_domain = re.search(r'^([^0-9]+)', sd_qid_question).group(1).strip()
                qid = re.search(r'([0-9]+.*?)(?= \()', sd_qid_question).group(1)
                lang, word = prediction[1].split(': ')
                question = re.search(r'\((.*)\)', sd_qid_question).group(1)
                key = f'{qid}#{lang}#{word}'
                f.write(key + ',')  # key
                f.write(qid + ',')  # qid
                f.write('"' + semantic_domain + '",')  # semantic domain
                f.write(lang + ',')  # language
                f.write(word + ',')  # word
                f.write('"' + question + '",')  # question
                f.write(str(prediction[2]) + ',')  # score
                f.write(f'{prediction[3]:.2f},')  # raw score
                f.write(str(prediction[4]) + '\n')  # is correct
                # f.write('"' + str(prediction[5]) + '"\n')  # similar words
                # f.write('"' + str(prediction[6]) + '"\n')  # explanation
        print('Done')

    @staticmethod
    def find_ideal_threshold(target, predicted):
        # https://stackoverflow.com/a/56204455/8816968
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]

    @staticmethod
    def find_ideal_threshold_to_maximize_f1_score(target, predicted):
        thresholds = [0.0, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 0.99,
                      1.0]  # np.linspace(0.00, 1.00, int((1.00 - 0.00) / 0.01) + 1)
        func = partial(compute_f1_score_with_threshold, target, predicted)
        # todo: try to use Pool to parallelize
        f1_scores = tqdm(map(func, thresholds),
                         desc='Finding ideal threshold to maximize F1 score',
                         total=len(thresholds))
        thresholds_and_f1_scores = list(zip(thresholds, f1_scores))
        thresholds_and_f1_scores.sort(key=lambda x: x[1], reverse=True)
        print('Top 3 thresholds and their F1 scores:', thresholds_and_f1_scores[:3])
        return thresholds_and_f1_scores[0][0]

    @staticmethod
    def find_ideal_threshold_to_maximize_recall_for_given_precision(target, predicted, target_precision):
        thresholds = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 0.99, 1.0]
        func = partial(compute_precision_and_recall_with_threshold, target, predicted)
        precisions_and_recall = tqdm(map(func, thresholds),
                                     desc=f'Finding ideal threshold to maximize recall for target precision {target_precision}',
                                     total=len(thresholds))
        precisions, recalls = zip(*precisions_and_recall)
        thresholds_precisions_recalls = list(zip(thresholds, precisions, recalls))

        # return the threshold for the highest recall above the target precision
        # if there is no such threshold, return the threshold for the highest precision
        thresholds_precisions_recalls.sort(key=lambda x: x[2], reverse=True)
        for threshold, precision, recall in thresholds_precisions_recalls:
            if precision >= target_precision:
                return threshold
        thresholds_precisions_recalls.sort(key=lambda x: x[1], reverse=True)
        return thresholds_precisions_recalls[0][0]

    @torch.no_grad()
    def evaluate(self, mask=None, plots=None, print_human_readable_predictions=False,
                 compute_additional_metrics=False, threshold=None, compute_ideal_threshold=True, num_frame=None,
                 print_highest_qid_correlations=False):
        assert ((threshold is not None and 0 <= threshold <= 1) or compute_ideal_threshold)
        if plots is None:
            plots = []
        if mask is None:
            print('WARNING: This evaluation is less expressive because it also includes the training set.')
            mask = torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.device)
        self.model.eval()
        y_hats = []
        ys = []
        node_idxs = []

        # eval_loader = LinkNeighborLoader(
        #     data=eval_data.to(self.device),
        #     num_neighbors=[-1, -1],
        #     edge_label_index=(
        #         self.TARGET_EDGE_TYPE, eval_data[self.TARGET_EDGE_TYPE].edge_label_index.to(self.device)),
        #     edge_label=eval_data[self.TARGET_EDGE_TYPE].edge_label.to(self.device),
        #     batch_size=self.config['batch_size'],  # number of edges per batch
        #     shuffle=True,
        # )
        eval_loaders = [NeighborLoader(
            self.data,
            num_neighbors=[-1],
            batch_size=self.config['batch_size'],
            input_nodes=mask.to(self.device) & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        ) for color_mask in self.color_masks if (mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        for eval_loader in tqdm(eval_loaders, desc='Evaluating...', total=len(eval_loaders)):
            for batch in eval_loader:
                x = batch.x.clone()
                edge_index = batch.edge_index
                edge_weight = batch.edge_weight
                x = x.to_dense()
                x[:batch.batch_size] = 0  # mask out the nodes to be predicted
                y_hat = self.model(x, edge_index, edge_weight)[
                        :batch.batch_size]
                y_hats.append(y_hat.detach().cpu())
                y = batch.y[:batch.batch_size]
                ys.append(y.detach().cpu())
                node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())

        y_hat = torch.cat(y_hats, dim=0)
        y = torch.cat(ys, dim=0)
        node_idxs = torch.cat(node_idxs, dim=0)
        assert y.sum() > 0  # at least one positive example
        assert (len(y) == len(y_hat))
        assert (len(y) == len(node_idxs))
        assert (len(y) == mask.sum())
        print(f'Computing evaluation metrics for {len(y)} words...')
        eval_loss = self.compute_loss_with_empty_questions(y_hat, y, node_idxs)
        soft_precision = self.compute_loss_with_empty_questions(y_hat, y, node_idxs,
                                                                soft_precision_function)
        soft_recall = self.compute_loss_with_empty_questions(y_hat, y, node_idxs,
                                                             soft_recall_function)
        f1_loss = -1.0  # f1_loss_function(y_hat, y)
        link_probs = y_hat.sigmoid()
        y_tensor = y.cpu()
        link_probs_tensor = link_probs.cpu()
        y = y.cpu().numpy()
        link_probs = link_probs.cpu().numpy()
        y_flat = y.flatten()
        link_probs_flat = link_probs.flatten()
        ideal_threshold = -1

        if compute_ideal_threshold:
            ideal_threshold = self.find_ideal_threshold_to_maximize_recall_for_given_precision(y_flat,
                                                                                               link_probs_flat,
                                                                                               0.95)  # self.find_ideal_threshold(y_flat, y_hat_flat)

        if threshold is None:
            threshold = ideal_threshold

        false_positives = false_negatives = precision = recall = acc = f1 = -1.0
        if compute_additional_metrics:
            print(f'Eval loss: {eval_loss.item():.2f}')
            raw_num_predictions = len(y_flat)
            all_predictions = torch.ones_like(y_tensor)
            num_predictions, _ = self.compute_metric_with_empty_questions(all_predictions, node_idxs)
            print(f'Samples: {mask.sum()} words, {num_predictions} predictions, raw: {raw_num_predictions}')
            print(f'Ideal threshold: {ideal_threshold:.2f}')
            print(f'Used threshold: {threshold:.2f}')

            candidates = torch.logical_and(link_probs_tensor >= threshold, y_tensor == 0)
            raw_false_positives = torch.sum(candidates).item()
            false_positives, false_positives_tensor = self.compute_metric_with_empty_questions(
                candidates, node_idxs)
            print(
                f'False positives: {false_positives} ({false_positives / num_predictions * 100:.2f} %), raw: {raw_false_positives}')

            candidates = torch.logical_and(link_probs_tensor < threshold, y_tensor == 1)
            raw_false_negatives = torch.sum(candidates).item()
            false_negatives, _ = self.compute_metric_with_empty_questions(candidates, node_idxs)
            print(
                f'False negatives: {false_negatives} ({false_negatives / num_predictions * 100:.2f} %), raw: {raw_false_negatives}')

            candidates = torch.logical_and(link_probs_tensor >= threshold, y_tensor == 1)
            raw_true_positives = torch.sum(candidates).item()
            true_positives, _ = self.compute_metric_with_empty_questions(candidates, node_idxs)
            print(
                f'True positives: {true_positives} ({true_positives / num_predictions * 100:.2f} %), raw: {raw_true_positives}')

            candidates = torch.logical_and(link_probs_tensor < threshold, y_tensor == 0)
            raw_true_negatives = torch.sum(candidates).item()
            true_negatives, _ = self.compute_metric_with_empty_questions(candidates, node_idxs)
            print(
                f'True negatives: {true_negatives} ({true_negatives / num_predictions * 100:.2f} %), raw: {raw_true_negatives}')

            # num_total_words = len(self.word_node_names) + self.num_removed_nodes
            # num_filtered_words = len(self.word_node_names)
            # pre_alignment_factor = num_total_words / num_filtered_words
            # print(f'Pre-alignment factor: {pre_alignment_factor:.2f} (total words: {num_total_words}, filtered now: {num_filtered_words})')
            # pre_alignment_recall = recall * pre_alignment_factor

            raw_precision = raw_true_positives / (
                    raw_true_positives + raw_false_positives) if raw_true_positives + raw_false_positives > 0 else 0.0
            precision = true_positives / (
                    true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
            print(f'Precision: {precision:.2f}, raw: {raw_precision:.2f}')

            raw_recall = raw_true_positives / (
                    raw_true_positives + raw_false_negatives) if raw_true_positives + raw_false_negatives > 0 else 0.0
            recall = true_positives / (
                    true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
            print(f'Recall: {recall:.2f}, raw: {raw_recall:.2f}')

            raw_f1 = 2 * (raw_precision * raw_recall) / (
                    raw_precision + raw_recall) if raw_precision + raw_recall > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            print(f'F1: {f1:.2f}, raw: {raw_f1:.2f}')

            if 'explain false positives' in plots:
                self.explain_false_positives(false_positives_tensor, node_idxs)

        if 'roc_curve' in plots:
            y_probas = np.array([1 - link_probs_flat, link_probs_flat]).T
            if y_flat.sum() > 0:
                skplt.metrics.plot_roc(y_flat, y_probas)
            plt.show()

        if 'probs' in plots:
            # plot distribution of prediction probabilities with log scale
            plt.hist(y_hat.flatten(), bins=100, log=True)
            plt.show()
            plt.hist(link_probs_flat, bins=100, log=True)
            plt.show()

        if 'weights' in plots:
            self.model.visualize_weights('after/during training, ' + self.wandb_original_run_name, num_frame, eval_loss,
                                         self.config['plot_path'])

        if print_highest_qid_correlations:
            self.model.print_highest_qid_correlations(self.sd_node_names)

        if print_human_readable_predictions:
            self.print_human_readable_predictions(y, y_hat, link_probs, None, threshold, word_idx_map=node_idxs)

        return eval_loss, ideal_threshold, false_positives, false_negatives, precision, recall, acc, f1, f1_loss, soft_precision, soft_recall

    @staticmethod
    def convert_edge_index_to_set_of_tuples(edge_index):
        return {(u, v) for u, v in edge_index.transpose(0, 1).tolist()}

    def save_model(self, path):
        path = os.path.join(self.config['model_path'], path)
        print(f'Saving model to {path}')
        torch.save(self.model.state_dict(), path)

    def update_original_run_name(self, model_path):
        # extract original name from path (e.g. 'model_fanciful-gorge-55_epoch417_val-loss0.917_checkpoint.bin' -> 'fanciful-gorge-55')
        self.wandb_original_run_name = re.search(r'(?<=model_)(.*?)(?=[_\.])', os.path.basename(model_path)).group(1)

    def load_model(self, path):
        path = os.path.join(self.config['model_path'], path)
        print(f'Loading model from {path}')
        self.update_original_run_name(path)

        self.init_model(self.data)
        self.model.load_state_dict(torch.load(path,
                                              map_location=None if self.device == 'cuda' else torch.device('cpu')))
        self.model.visualize_weights('loaded, ' + self.wandb_original_run_name)
        self.model.to(self.device)
        self.model.eval()
        print('Done loading model.')

    def create_gif(self):
        assert (len(self.model.plot_file_paths) > 0)

        # Create a gif from the plots
        path = os.path.join(self.config['plot_path'], f'0_plot_weights_{wandb.run.name}.gif')
        print('Creating gif at ' + path)
        with imageio.get_writer(path, mode='I') as writer:
            for file_path in self.model.plot_file_paths:
                image = imageio.v2.imread(file_path)
                writer.append_data(image)

        print('Done creating gif.')

    @torch.no_grad()
    def predict(self, test_word, threshold):
        # clone the data to avoid changing the original data
        pred_data = self.data.clone()
        pred_data.to(self.device)

        # build mask for test_word
        test_word_idx = self.word_node_idx_by_name[test_word]
        mask = torch.zeros(len(pred_data.x), dtype=torch.bool)
        mask[test_word_idx] = True
        mask = mask.to(self.device)

        # mask out the current word
        # pred_data.x = pred_data.x.to_dense()
        pred_data.x[mask] = 0

        y_hat = self.model(pred_data.x, pred_data.edge_index, pred_data.edge_weight)
        link_probs = y_hat.sigmoid()

        # print all positive predictions
        self.print_human_readable_predictions(None, y_hat, link_probs, mask, threshold)
        print('\n')

    @torch.no_grad()
    def predict_for_languages(self, langs, threshold):
        # clone the data to avoid changing the original data
        pred_data = self.data.clone()
        pred_data.to(self.device)

        # build mask for languages
        mask = torch.zeros(len(pred_data.x), dtype=torch.bool)
        for lang in langs:
            mask[self.word_node_idxs_by_lang[lang]] = True
        mask = mask.to(self.device)

        node_loaders = [NeighborLoader(
            pred_data,
            num_neighbors=[-1],
            batch_size=self.config['batch_size'],
            input_nodes=mask.to(self.device) & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        ) for color_mask in self.color_masks if (mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        ys = []
        y_hats = []
        node_idxs = []
        for node_loader in node_loaders:
            for batch in node_loader:
                batch.to(self.device)
                x = batch.x.clone()
                y = batch.y[:batch.batch_size]
                ys.append(y.detach().cpu())
                # x = x.to_dense()
                x[:batch.batch_size] = 0  # mask out the nodes to be predicted
                y_hat = self.model(x, batch.edge_index, batch.edge_weight)
                y_hat = y_hat[:batch.batch_size]
                y_hats.append(y_hat.detach().cpu())
                node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())

        torch.cuda.empty_cache()
        y = torch.cat(ys, dim=0)
        y_hat = torch.cat(y_hats, dim=0)
        node_idxs = torch.cat(node_idxs, dim=0)

        link_probs = y_hat.sigmoid()
        assert (mask.sum() == len(y_hat))
        assert (mask.sum() == len(node_idxs))
        assert (len(node_idxs) == len(set(node_idxs)))  # assert that every idx in node_idxs appears exactly once

        # print all positive predictions
        positive_prediction_mask = link_probs.sum(dim=1) >= threshold
        self.print_human_readable_predictions(y, y_hat, link_probs, positive_prediction_mask, threshold,
                                              max_print_words=int(1e10),
                                              save_to_csv_path=f'data/6_results/predicted_sds_{langs}_{self.wandb_original_run_name}.csv',
                                              word_idx_map=node_idxs, max_print_total=int(1e10))
        print('\n')

    def explain_false_positive(self, node_idx, qid_idx, fp_node_name_by_qid_by_node=None):
        node_name = self.word_node_names[node_idx]
        qid_name = self.sd_node_names[qid_idx]
        # neighbors = list(self.G.neighbors(node_name))
        node_mask = torch.zeros(len(self.data.x), dtype=torch.bool)
        node_mask[node_idx] = True

        # select only the current node
        node_loader = NeighborLoader(
            self.data.to(self.device),
            num_neighbors=[-1],
            batch_size=self.config['batch_size'],
            input_nodes=node_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        )

        assert (len(node_loader) == 1)
        batch = next(iter(node_loader))
        assert (batch.batch_size == 1)

        batch.to(self.device)
        x = batch.x.clone()
        # x = x.to_dense()
        x[:batch.batch_size] = 0  # mask out the node to be predicted
        y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)
        y_hat = y_hat[:batch.batch_size].detach().cpu()
        base_prediction = y_hat[0][qid_idx]

        qid_diff_by_neighbor = {}
        num_neighbors = len(batch.x) - batch.batch_size
        for i in range(num_neighbors):
            # mask out each neighbor individually and predict
            neighbor_idx = batch.batch_size + i
            x_neighbor = batch.x.clone()
            # x = x.to_dense()
            x_neighbor[neighbor_idx] = 0  # mask out the neighbor # continue here: fix this
            y_hat = self.model(x_neighbor, batch.edge_index, batch.edge_weight)

            y_hat = y_hat[:batch.batch_size].detach().cpu()
            qid_diff = base_prediction - y_hat[0][qid_idx]
            original_neighbor_idx = batch.n_id[neighbor_idx].detach().cpu()
            neighbor_name = self.word_node_names[original_neighbor_idx]
            qid_diff_by_neighbor[neighbor_name] = qid_diff.item()
            if fp_node_name_by_qid_by_node is not None and qid_diff.item() > 0:
                fp_node_name_by_qid_by_node[qid_name][neighbor_name].add(node_name)

        assert fp_node_name_by_qid_by_node is None or len(fp_node_name_by_qid_by_node[qid_name]) > 0
        assert (abs(base_prediction - sum(qid_diff_by_neighbor.values()) - self.model.conv1.bias[qid_idx]) < 1e-2)

        # sort by difference
        qid_diff_by_neighbor = {k: round(v * 10) / 10 for k, v in
                                sorted(qid_diff_by_neighbor.items(), key=lambda item: item[1], reverse=True)}
        return qid_diff_by_neighbor

    def get_edge_color(self, node_name, qid_name):
        if self.df_additional_words is None:  # lazy load
            path = 'data/4_semdoms/additional_words.xlsx'
            self.df_additional_words = pd.read_excel(path)
            self.df_additional_words = self.df_additional_words.set_index('key')

        qid = re.search(r'([0-9]+.*?)(?= \()', qid_name).group(1)
        key = qid + '#' + node_name.replace(': ', '#')
        is_manually_verified = False
        translation = ''
        word_has_gpt4_answer = False

        if key in self.df_additional_words.index:
            is_manually_verified = self.df_additional_words.loc[key]['manually verified'] == 1
            translation = f' ({self.df_additional_words.loc[key]["deu translation"]})'
            translation = '' if translation == ' (nan)' else translation
            if self.df_additional_words.loc[key]['GPT-4 answer'] == 1:
                color = 'orange'  # GPT-4 says the word belongs to the QID
                word_has_gpt4_answer = True
            elif self.df_additional_words.loc[key]['GPT-4 answer'] == 0:
                color = 'red'  # GPT-4 says the word does not belong to the QID
                # assert node_name not in self.word_node_names_by_sd_name[qid_name]
                if node_name in self.word_node_names_by_sd_name[qid_name]:
                    print(
                        f'WARNING: {node_name} is marked as a false positive, but it is in the GT dict for {qid_name}.')
                word_has_gpt4_answer = True

        if not word_has_gpt4_answer:
            if node_name in self.word_node_names_by_sd_name[qid_name]:
                color = 'green'  # the GT dict says the word belongs to the QID
                is_manually_verified = True
            else:
                color = 'grey'  # unknown word

        bold_style = 'font-weight:bold;' if is_manually_verified else ''
        return color, key, translation, bold_style

    def explain_false_positives(self, false_positives, node_idxs):
        # for each false positive (i.e., a node with an incorrectly predicted semantic domain question), print all its neighbors
        false_positives = false_positives.nonzero(as_tuple=False)
        total = len(false_positives)
        fp_node_names_by_qid_by_node = defaultdict(lambda: defaultdict(set))

        for node_idx, qid_idx in tqdm(false_positives, total=total, desc='Explaining false positives'):
            original_node_idx = node_idxs[node_idx]
            qid_diff_by_neighbor = self.explain_false_positive(original_node_idx, qid_idx, fp_node_names_by_qid_by_node)
            # print(f'qid_diff_by_neighbor: {qid_diff_by_neighbor}')

        # in each qid, sort by number of false positives
        for qid_name in fp_node_names_by_qid_by_node:
            fp_node_names_by_qid_by_node[qid_name] = {k: v for k, v in
                                                      sorted(fp_node_names_by_qid_by_node[qid_name].items(),
                                                             key=lambda item: len(item[1]), reverse=True)}
        # sort qids by sum of false positives
        fp_node_names_by_qid_by_node = {k: v for k, v in
                                        sorted(fp_node_names_by_qid_by_node.items(),
                                               key=lambda item: sum(len(x) for x in item[1].values()), reverse=True)}

        # Open an HTML file to write
        path = f'data/6_results/fp_node_name_by_qid_by_node_{self.wandb_original_run_name}.html'
        with open(path, 'w') as f:
            # Write the HTML header
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Results</title>
                <style>
                    .colorButton {
                        border: none;
                        text-align: center;
                        display: inline-block;
                        cursor: pointer;
                        border-radius: 1em;
                    }
                </style>
                
                <script>
                    const colors = ['lightgray', 'lightgreen', 'pink'];
                    const sound = new Audio("sounds/mixkit-cool-interface-click-tone-2568.wav");

                    function changeColor(clickedButton) {
                        // Get all buttons of the given group type
                        groupType = clickedButton.getAttribute('data-group');
                        var buttons = document.querySelectorAll('button[data-group="' + groupType + '"]');
                        
                        // Change the color of the clicked button
                        var currentColor = clickedButton.style.backgroundColor;
                        var index = colors.indexOf(currentColor);
                        index = (index + 1) % colors.length;
                        clickedButton.style.backgroundColor = colors[index];
                        sound.play();
                        sound.currentTime = 0;
                        
                        // Change the color of the other buttons
                        for (var i = 0; i < buttons.length; i++) {
                            button = buttons[i];
                            button.style.backgroundColor = clickedButton.style.backgroundColor;    
                        }
                        
                        // Log the group type and the clicked button's status
                        console.log(groupType, clickedButton.style.backgroundColor);
                    }
                    
                    function printAllActiveButtons() {
                        // Get all data groups
                        var dataGroups = document.querySelectorAll('button[data-group]');
                        
                        // Print the group type and the color of one of its buttons
                        output = ''
                        for (var i = 0; i < dataGroups.length; i++) {
                            var dataGroup = dataGroups[i];
                            
                            // Get the first button of the group
                            var buttons = document.querySelectorAll('button[data-group="' + dataGroup.getAttribute('data-group') + '"]');
                            var button = buttons[0];
                            
                            // Print the group type and the color of the first button (if it is not lightgray)
                            color = button.style.backgroundColor;
                            if (color != 'lightgray') {
                                output += dataGroup.getAttribute('data-group')
                                output += ': '
                                output += button.style.backgroundColor
                                output += ', '
                            }
                        }
                        console.log(output);
                    } 
                </script>
            </head>
            <body>
                <button onclick="printAllActiveButtons()">Print All Active Buttons</button>
            """)

            # Start printing the false positives
            print(f'Saving false positives to {path}...')
            for qid_name in fp_node_names_by_qid_by_node:
                f.write(f'<h1>{qid_name[5:]}</h1>\n')
                for node_name in fp_node_names_by_qid_by_node[qid_name]:
                    # Print node_name with the color
                    word_color, key, translation, bold_style = self.get_edge_color(node_name, qid_name)
                    background_color = '#f0f0f0' if bold_style != '' else 'lightgray'
                    key = 'source-word#' + key
                    f.write(
                        f'  <p><button class="colorButton" data-group="{key}" onclick="changeColor(this)" style="background-color:{background_color}">'
                        f'<span style="color:{word_color};{bold_style}">{node_name}</span>{translation}'
                        f'</button> → ')

                    # Iterate over the words in the node_names, and print each with its color
                    for word in fp_node_names_by_qid_by_node[qid_name][node_name]:
                        word_color, key, translation, bold_style = self.get_edge_color(word, qid_name)
                        background_color = '#f0f0f0' if bold_style != '' else 'lightgray'
                        key = 'target-word#' + key
                        f.write(
                            f'<button class="colorButton" data-group="{key}" onclick="changeColor(this)" style="background-color:{background_color}">'
                            f'<span style="color:{word_color};{bold_style}">{word}</span>{translation}'
                            f'</button> ')

                    f.write('</p>\n')

            f.write('<br/>')

            # Close the HTML tags
            f.write("""
            </body>
            </html>
            """)
        print('Done.')

    def init_model(self, data):
        self.model = Model(
            in_channels=data.num_features,
            out_channels=data.y.size(1),
            bias=self.config['bias'],
        )

        self.model.to(self.device)

    def test(self, print_human_readable_predictions=False, threshold=None, eval_train_set=False,
             compute_ideal_threshold=True, plots=None):
        plots = [] if plots is None else plots

        print('\nValidation set performance:')
        _, ideal_threshold, _, _, _, _, _, _, _, _, _ = self.evaluate(
            self.data.val_mask, plots, print_human_readable_predictions, True, threshold,
            compute_ideal_threshold=compute_ideal_threshold, print_highest_qid_correlations=False)  # True)

        for lang in self.gt_langs:
            print(f'\nValidation set performance for {lang}:')
            lang_mask = torch.zeros(len(self.data.x), dtype=torch.bool)
            lang_mask[self.word_node_idxs_by_lang[lang]] = True
            lang_mask = lang_mask.to(self.device)
            lang_mask &= self.data.val_mask.to(self.device)

            _, _, _, _, _, _, _, _, _, _, _ = self.evaluate(
                lang_mask, [], False, True, ideal_threshold if threshold is None else threshold,
                compute_ideal_threshold=False)

        print('\nTest set performance:')
        test_loss, _, false_positives, false_negatives, test_precision, test_recall, acc, test_f1, f1_loss, _, _ = self.evaluate(
            self.data.test_mask, [], print_human_readable_predictions, True,
            ideal_threshold if threshold is None else threshold, compute_ideal_threshold=False)

        if wandb.run is not None:
            wandb.log({"test loss": test_loss, "test F1 loss": f1_loss,
                       "val ideal threshold": ideal_threshold,
                       "false positives": false_positives, "false negatives": false_negatives,
                       "precision": test_precision,
                       "recall": test_recall, "ACC": acc, "F1": test_f1})
            wandb.watch(self.model)

        if eval_train_set:
            print('\nTrain set performance:')
            self.evaluate(self.data.train_mask, [], print_human_readable_predictions, True,
                          ideal_threshold if threshold is None else threshold, compute_ideal_threshold=False)

        return test_loss, ideal_threshold, test_f1, test_precision

    def split_dataset(self):
        split = T.RandomNodeSplit(
            num_val=0.1,
            num_test=0.1,
        )
        self.data = split(self.data)
        print(self.data)

    def partition_graph(self):
        # solve n-color problem for multiple hops
        hop_graph = self.G.copy()

        # remove sd nodes
        hop_graph.remove_nodes_from(self.sd_node_names)

        print('Partitioning graph...')
        color_by_node = nx.greedy_color(hop_graph)
        print('Done partitioning graph')

        # create a list of bit masks (one for each color)
        self.color_masks = []
        for color in range(max(color_by_node.values()) + 1):
            mask = torch.zeros(hop_graph.number_of_nodes(), dtype=torch.bool)
            for node, node_color in color_by_node.items():
                if node_color == color:
                    node_idx = self.word_node_idx_by_name[node]
                    mask[node_idx] = True

            self.color_masks.append(mask.to(self.device))

        # assert that the masks are complete
        assert (sum([m.sum() for m in self.color_masks]).item() == len(self.color_masks[0]))

    def plot_class_counts(self):
        print('Plotting class counts...')

        # count how often each class occurs in self.data
        class_counts = torch.sum(self.data.y, dim=0).cpu().numpy()
        class_counts = {f'{self.sd_node_names[i]}': count for i, count in enumerate(class_counts)}

        # get the actual QIDs
        class_counts = {re.search(r'([0-9]+.*?)(?= \()', key).group(1): count for key, count in class_counts.items()}

        # plot the class counts
        plt.figure(figsize=(20, 10))
        plt.tight_layout()
        plt.title('Class counts')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xticks(rotation=-90)
        plt.show()

    def save_dataset(self):
        # save the dataset with cPickle
        path = f'data/9_datasets/{self.wandb_original_run_name}.cpickle'
        if os.path.exists(path):
            print(f'File {path} already exists, skipping saving the dataset.')
            return

        print(f'Saving dataset to {path}')
        with open(path, 'wb') as f:
            cPickle.dump((self.data, self.sd_node_idx_by_name, self.word_node_idx_by_name, self.sd_node_names,
                          self.word_node_names, self.word_node_idxs_by_lang, self.word_node_names_by_sd_name,
                          self.forbidden_neg_train_edges, self.train_data, self.val_data, self.test_data,
                          self.color_masks), f)
        print('Done.')

    def load_dataset(self):
        # load the dataset with cPickle
        path = f'data/9_datasets/{self.wandb_original_run_name}.cpickle'
        print(f'Loading dataset from {path}')
        with open(path, 'rb') as f:
            gc.disable()  # disable garbage collection to speed up loading
            self.data, self.sd_node_idx_by_name, self.word_node_idx_by_name, self.sd_node_names, self.word_node_names, \
                self.word_node_idxs_by_lang, self.word_node_names_by_sd_name, self.forbidden_neg_train_edges, \
                self.train_data, self.val_data, self.test_data, self.color_masks = cPickle.load(f)
            gc.enable()  # enable garbage collection again
        print('Done.')

    def build_dataset(self, additional_lang=None):
        self.build_network()  # also contains languages without gt sds
        subgraph = self.G.subgraph([n for n in self.G.nodes if
                                    self.G.nodes[n]['lang'] in self.gt_langs.union(
                                        {'semantic_domain_question',
                                         additional_lang} if additional_lang is not None else {
                                            'semantic_domain_question'})])
        self.G, complete_graph = subgraph, self.G

        self.sd_node_names = [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "semantic_domain_question"]
        self.word_node_names = [node for node in self.G.nodes if
                                self.G.nodes[node]["lang"] != "semantic_domain_question"]
        word_nodes_set = set(self.word_node_names)

        # Create dictionaries to map nodes to indices
        self.sd_node_idx_by_name = {node: i for i, node in enumerate(self.sd_node_names)}
        self.word_node_idx_by_name = {node: i for i, node in enumerate(self.word_node_names)}

        self.word_node_names_by_sd_name = {sd_node: list(self.G.neighbors(sd_node)) for sd_node in self.sd_node_names}

        alignment_edges = [(self.word_node_idx_by_name[edge[0]], self.word_node_idx_by_name[edge[1]]) for edge in
                           self.G.edges if edge[0] in word_nodes_set and edge[1] in word_nodes_set]

        # Save the lang information, use node index as key
        lang_by_word_node_idx = {self.word_node_idx_by_name[node]: self.G.nodes[node]["lang"] for node in
                                 self.word_node_names}

        self.word_node_idxs_by_lang = {
            lang: [self.word_node_idx_by_name[node] for node in self.word_node_names if
                   self.G.nodes[node]["lang"] == lang] for
            lang
            in set(lang_by_word_node_idx.values())}
        num_words_by_lang = {lang: len(self.word_node_idxs_by_lang[lang]) for lang in self.word_node_idxs_by_lang}
        print(num_words_by_lang)
        print(sum(num_words_by_lang.values()))

        self.convert_empty_questions_by_lang_to_tensors()

        if not torch.cuda.is_available():  # hack to run this only on my laptop
            self.plot_subgraph(self.G, 'eng: cousin')
            self.plot_subgraph(self.G, 'por: hábito')
            self.plot_subgraph(complete_graph, 'deu: läutern')

            # # plot neighborhoods
            # self.plot_subgraph(self.G, random.choice(
            #     [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "ind"]))
            # self.plot_subgraph(self.G, random.choice(
            #     [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "fra"]))
            # self.plot_subgraph(complete_graph, random.choice(
            #     [node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == "deu"]))
            # self.plot_subgraph(complete_graph, random.choice(
            #     [node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == "gej"]))

        # Create language feature (for word nodes)
        lang_feature = torch.zeros(len(self.word_node_names), len(self.all_langs))
        for node in self.word_node_names:
            lang = complete_graph.nodes[node]["lang"]
            lang_feature[self.word_node_idx_by_name[node]][self.all_langs.index(lang)] = 1
            # lang_feature[self.word_node_idx_by_name[node]][0] = 1

        # Create the node degree feature
        degree_feature = torch.zeros(len(self.word_node_names), 1)
        for node in self.word_node_names:
            degree_feature[self.word_node_idx_by_name[node]] = self.G.degree(node)

        # Create the weighted node degree feature
        weighted_degree_feature = torch.zeros(len(self.word_node_names), 1)
        for node in self.word_node_names:
            weighted_degree_feature[self.word_node_idx_by_name[node]] = self.G.degree(node, weight='weight')

        # Create QID feature
        indices = []
        values = []
        for node in tqdm(self.word_node_names, desc='Creating QID feature', total=len(self.word_node_names)):
            # set a 1 for each QID to which the word is connected
            for neighbor in self.G.neighbors(node):
                if neighbor in self.sd_node_names:
                    indices.append([self.word_node_idx_by_name[node], self.sd_node_idx_by_name[neighbor]])
                    values.append(1)
        qid_feature = torch.sparse_coo_tensor(torch.tensor(indices).transpose(0, 1), torch.tensor(values),
                                              size=(len(self.word_node_names), len(self.sd_node_names))).to_dense()

        # Create the QID count feature (count qids in qid_feature)
        qid_count_feature = torch.zeros(len(self.word_node_names), 1)
        for node in self.word_node_names:
            qid_count_feature[self.word_node_idx_by_name[node]] = torch.sum(
                qid_feature[self.word_node_idx_by_name[node]])

        # concatenate the features
        # x = qid_feature
        # qid_count_feature.to_sparse_coo<()
        # degree_feature.to_sparse_coo()
        # weighted_degree_feature.to_sparse_coo()
        #
        # concat_dim = 0
        # new_indices = torch.cat([i1, i2 + (i1.max(dim=1).values[concat_dim] + 1)], dim=1)
        # new_values = torch.cat([v1, v2], dim=0)
        #
        x = torch.cat((qid_count_feature, degree_feature, weighted_degree_feature, qid_feature), dim=1)

        alignment_edge_index = torch.tensor(alignment_edges).transpose(0, 1)
        alignment_edge_weight = torch.tensor(
            [self.G.edges[(self.word_node_names[edge[0]], self.word_node_names[edge[1]])].get("weight",
                                                                                              0)
             for edge in alignment_edges]).type(torch.FloatTensor)

        self.data = Data(x=x, edge_index=alignment_edge_index, y=qid_feature, edge_weight=alignment_edge_weight)

        self.data = T.ToUndirected()(self.data)
        self.data = self.data.to(self.device)
        print(self.data)
        # self.plot_class_counts()

        self.split_dataset()
        self.partition_graph()
        self.save_dataset()

        print("Creating train loaders...")
        self.train_loaders = [NeighborLoader(
            self.data,
            num_neighbors=[-1],
            # [-1] vs [-1, -1] makes a difference if normalization=1 because it changes the neighbor's edge weights
            batch_size=self.config['batch_size'],
            input_nodes=self.data.train_mask.to(self.device) & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        ) for color_mask in self.color_masks if
            (self.data.train_mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        self.criterion = f1_loss_function

    def make_predictions(self):
        path_1 = 'model_lemon-lion-389_precision0.90_F10.43_th-1.00_deu+eng+fra+hin+ind+mal+nep+por+spa+swa.bin'  # 'model_fancy-blaze-294_precision0.71_F10.28_th-1.00_deu+eng+fra+hin+ind+mal+nep+por+spa+swa.bin' # 'model_denim-valley-216_precision0.91_F10.54_th-1.00_deu+eng+fra+hin+ind+mal+nep+por+spa+swa.bin'  # 'model_dauntless-plant-200_precision0.77_F10.46_th-1.00_deu+eng+fra+gej+hin+ind+mal+meu+nep+por+spa+swa+tpi+yor.bin' # 'model_breezy-pine-160.bin' # 'model_smart-frog-156_precision0.77_F10.45_th-1.00_deu+eng+fra+gej+hin+ind+mal+meu+nep+por+spa+swa+tpi+yor.bin'  # 'model_morning-breeze-153_precision0.75_F10.43_th-1.00_deu+eng+fra+gej+hin+ind+mal+meu+nep+por+spa+swa+tpi+yor.bin' # 'model_worldly-silence-151_precision0.68_F10.39_th-1.00_cmn+deu+eng+fra+gej+hin+ind+mal+meu+mya+nep+pes+por+spa+swa+tpi+urd+yor.bin' #'model_dulcet-mountain-132_precision0.46_F10.25_th-1.00_cmn+deu+eng+fra+gej+hin+ind+mal+meu+mya+nep+pes+por+spa+swa+tpi+urd+yor.bin' # model_fast-dew-9_precision0.47_F10.21_th-1.00_cmn+deu+eng+fra+gej+hin+ind+mal+meu+mya+nep+pes+por+spa+swa+tpi+urd+yor_precision0.51_F10.25_th-1.00_cmn+deu+eng+fra+gej+hin+ind+mal+meu+mya+nep+pes+por+spa+swa+tpi+urd+yor.bin'  # 'model_proud-pond-143.bin'
        self.update_original_run_name(path_1)
        # self.build_dataset()
        self.build_dataset(additional_lang='deu')
        # self.load_dataset() # crashes for 'deu'
        self.load_model(path_1)
        threshold = 1.0
        # self.predict('por: hábito', threshold)
        self.test(threshold=threshold, compute_ideal_threshold=False)
        self.evaluate(plots=['explain false positives'], threshold=threshold,
                      compute_ideal_threshold=False, compute_additional_metrics=True)

        # self.build_dataset(additional_lang='deu')
        self.predict_for_languages(['deu'],
                                   threshold)  # , 'yor', 'gej', 'tpi', 'meu'], threshold)

        self.build_dataset()
        self.predict_for_languages(['eng', 'fra', 'ind', 'por', 'swa', 'spa', 'hin', 'mal', 'nep'],
                                   threshold)  # , 'yor', 'gej', 'tpi', 'meu'], threshold)

    def run_gnn_pipeline(self):
        self.build_dataset()
        self.init_model(self.data)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config[
            'learning_rate'], weight_decay=self.config['weight_decay'])
        assert (self.data.is_undirected())

        self.train_link_predictor()

        # path_1 = 'model_charmed-mountain-56_epoch32_val-loss0.935_checkpoint.bin' # 'model_fast-dew-9_precision0.47_F10.21_th-1.00_cmn+deu+eng+fra+gej+hin+ind+mal+meu+mya+nep+pes+por+spa+swa+tpi+urd+yor_precision0.51_F10.25_th-1.00_cmn+deu+eng+fra+gej+hin+ind+mal+meu+mya+nep+pes+por+spa+swa+tpi+urd+yor.bin'  # 'model_proud-pond-143.bin'
        # self.load_model(path_1)
        path_1 = f"model_{wandb.run.name}.bin"
        self.save_model(path_1)
        test_loss, ideal_threshold, test_f1, test_precision = self.test(eval_train_set=True, threshold=1.00,
                                                                        compute_ideal_threshold=False,
                                                                        plots=['weights'])
        if "th-" not in path_1:
            path_2 = path_1[
                     :-4] + f"_precision{test_precision:.2f}_F1{test_f1:.2f}_th{ideal_threshold:.2f}_{'+'.join(self.dc.target_langs)}.bin"
            self.save_model(path_2)
            os.remove(os.path.join(self.config['model_path'], path_1))


if __name__ == '__main__':
    print("GNN setup started")

    project_name = "GCN node prediction (link semantic domains to target words), 18 langs (nep) dataset"
    wandb.init(
        project=project_name,
        config={
            "batch_size": 6000,  # 10000 # higher batch size --> higher recall
            "epochs": 1000,

            "bias": True,  # works much better than False
            "min_alignment_count": 1,  # 4,
            "min_edge_weight": 0.2,
            "patience": 5,
            "warm_up": 30,  # 10,

            "optimizer": "adam",
            "learning_rate": 0.05,  # 0.005,
            "weight_decay": 0.0,  # 0.005,
            "model_path": "data/3_models/",
            "plot_path": "data/8_plots/",
        })

    dc = LinkPredictionDictionaryCreator(['bid-eng-web', 'bid-fra-fob', 'bid-ind', 'bid-por', 'bid-swa', 'bid-spa',
                                          'bid-hin', 'bid-mal', 'bid-nep', 'bid-deu'])
    # 'bid-yor', 'bid-tpi', 'bid-meu', 'bid-gej'])
    lp = LinkPredictor(dc, {'eng', 'fra', 'ind', 'por', 'swa', 'spa', 'hin', 'mal', 'nep'}
                       , wandb.config,
                       graph_path=f'data/7_graphs/graph-nep_min_alignment_count_{wandb.config["min_alignment_count"]}_min_edge_weight_{wandb.config["min_edge_weight"]}_additional_words_20.cpickle')  # --> check that right code gets executed (training instead of predicting)
    # ignored_langs={'pes', 'urd', 'cmn', 'mya'})

    # lp.run_gnn_pipeline()
    lp.make_predictions()
    print("GNN pipeline finished")
