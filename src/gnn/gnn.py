import gc
import json
import math
import os
import pprint
import re
from collections import defaultdict, deque
from functools import partial
from math import sqrt

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
from torch_geometric.nn import GCNConv, DNAConv
from tqdm import tqdm

from src.dictionary_creator.link_prediction_dictionary_creator import LinkPredictionDictionaryCreator


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches
        # fig_width = 12 if columns==1 else 17 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        # 'backend': 'ps',
        'text.latex.preamble': '\\usepackage{gensymb}',
        'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 10,
        'lines.linewidth': 2.0,
        'axes.linewidth': 2.0,
        'font.size': 20,  # was 10
        'legend.fontsize': 10,  # was 10
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.markersize': 2,
        # 'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif'
    }

    plt.rcParams.update(params)


latexify(6, None)


# NOTES
# Der Unterschied zwischen edge_index und edge_label_index ist, dass edge_index die Kanten für den Encoder sind, während
# edge_label_index die Kanten für den Decoder sind.
# Der Encoder erstellt ein Node Embedding für jeden Knoten mit den Kanten aus edge_index. Daher darf er keine negativen Kanten bekommen.
# Der Decoder wird dann auf binärer Klassifikation von Kanten trainiert. Daher braucht er positive und negative Kanten.
# "edge_label_index will be used for the decoder to make predictions
# and edge_label will be used for model evaluation."
# edge_index contains no negative edges.
# Der Encoder sieht alle Kanten auf dem Plot.
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
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = GCNConv(in_channels, out_channels, add_self_loops=False, normalize=False, improved=True, bias=bias)
        # self.conv2 = GCNConv(out_channels, out_channels, add_self_loops=False, normalize=False, improved=True, bias=bias)
        # self.conv1 = DNAConv(in_channels, heads=1, add_self_loops=False, normalize=False, improved=True, bias=bias)

        # Smart initialize weight matrix
        print("Smart initializing weight matrix...")
        offset = in_channels - out_channels
        with torch.no_grad():
            for x in range(out_channels):
                # Set diagonal (identity matrix for the question feature)
                self.conv1.lin.weight[x][x + offset] = 50.0  # 2.0
                # # Set (weighted) node degree feature
                # self.conv1.lin.weight[x][0] = 0.01
                # self.conv1.lin.weight[x][1] = 0.01

        # Initialize bias vector
        print("Smart initializing bias vector...")
        with torch.no_grad():
            torch.nn.init.constant_(self.conv1.bias, -5.0)  # -3.0)

        # self.conv1 =  DNAConv(in_channels, add_self_loops=False, normalize=False, bias=bias)
        # head = 0
        #
        # # Smart initialize value weight matrix
        # print("Smart initializing value weight matrix...")
        # with torch.no_grad():
        #     for x in range(out_channels):
        #         # Set diagonal (identity matrix for the question feature)
        #         self.conv1.multi_head.lin_v.weight[head][x][x] = 50.0  # 2.0
        #         # # Set (weighted) node degree feature
        #         # self.conv1.lin.weight[x][0] = 0.01
        #         # self.conv1.lin.weight[x][1] = 0.01
        #
        # # Initialize value bias vector
        # print("Smart initializing value bias vector...")
        # with torch.no_grad():
        #     torch.nn.init.constant_(self.conv1.multi_head.lin_v.bias[head], -5.0)  # -3.0)
        #
        #
        # # Smart initialize key weight matrix
        # print("Smart initializing key weight matrix...")
        # with torch.no_grad():
        #     for x in range(out_channels):
        #         # Set diagonal (identity matrix for the question feature)
        #         self.conv1.multi_head.lin_k.weight[head][x][x] = 50.0  # 2.0
        #         # # Set (weighted) node degree feature
        #         # self.conv1.lin.weight[x][0] = 0.01
        #         # self.conv1.lin.weight[x][1] = 0.01
        #
        # # Initialize value bias vector
        # print("Smart initializing key bias vector...")
        # with torch.no_grad():
        #     torch.nn.init.constant_(self.conv1.multi_head.lin_k.bias[head], -5.0)  # -3.0)
        #
        #
        # # Smart initialize query weight matrix
        # print("Smart initializing query weight matrix...")
        # with torch.no_grad():
        #     for x in range(out_channels):
        #         # Set diagonal (identity matrix for the question feature)
        #         self.conv1.multi_head.lin_q.weight[head][x][x] = 50.0  # 2.0
        #         # # Set (weighted) node degree feature
        #         # self.conv1.lin.weight[x][0] = 0.01
        #         # self.conv1.lin.weight[x][1] = 0.01
        #
        # # Initialize value bias vector
        # print("Smart initializing query bias vector...")
        # with torch.no_grad():
        #     torch.nn.init.constant_(self.conv1.multi_head.lin_q.bias[head], -5.0)  # -3.0)

        # self.visualize_weights('initial')

    def forward(self, x, edge_index, edge_weight):
        z = self.conv1(x, edge_index, edge_weight)

        # x_all = x.view(-1, 1, self.in_channels)
        #
        # z = self.conv1(x_all, edge_index, edge_weight)
        #
        # # cut down to out_channels
        # z = z[:, -self.out_channels:]

        return z

    def visualize_weights(self, title, epoch=None, loss=None, plot_path=None, select_first_n_sdqs=50):
        # return # for DNAConv
        print('Visualizing weights...')

        # set size to 2000 x 2000 pixels
        plt.rcParams["figure.figsize"] = (20, 20)

        if type(self.conv1) is GCNConv:
            weight_matrix = self.conv1.lin.weight.detach().cpu()
            bias_vector = self.conv1.bias.detach().cpu() if self.conv1.bias is not None else None
            if select_first_n_sdqs is not None:
                offset = self.in_channels - self.out_channels
                weight_matrix = weight_matrix[:select_first_n_sdqs, :select_first_n_sdqs + offset]
                bias_vector = bias_vector[:select_first_n_sdqs] if bias_vector is not None else None
        elif type(self.conv1) is DNAConv:
            head = 0
            weight_matrix = self.conv1.multi_head.lin_v.weight[head].detach().cpu()
            bias_vector = self.conv1.multi_head.lin_v.bias[head].detach().cpu() if self.conv1.multi_head.lin_v.bias[
                                                                                       head] is not None else None
        else:
            raise NotImplementedError

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
        epoch = epoch if epoch is not None else 'epoch-unspecified'
        if plot_path is not None:
            file_name = f'plot_weight_matrix_{epoch}.png'
            file_path = os.path.join(plot_path, file_name)
            self.plot_file_paths.append(file_path)
            print(f'Saving plot to {file_path}')
            plt.savefig(file_path)
            plt.close()

            # Save weight_matrix_np to txt
            file_name = f'weight_matrix_{epoch}_{title}.txt'
            file_path = os.path.join(plot_path, file_name)
            print(f'Saving weight matrix to {file_path}')
            np.savetxt(file_path, weight_matrix.numpy(), fmt='%f')
        else:
            plt.show()

        if bias_vector is not None:
            bias_vector_np = bias_vector.numpy()

            # Add 1 dimension to the vectors
            bias_vector_np = np.expand_dims(bias_vector_np, axis=1)

            # bias has only one dimension, so we have to make it a 2D array to plot it
            bias_vector_np = np.repeat(bias_vector_np, repeats=10, axis=1)
            plt.imshow(bias_vector_np, cmap='bwr', interpolation='nearest'),  # norm=norm
            plt.colorbar()
            plt.title('Bias Vector Heatmap ' + title)
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            if plot_path is not None:
                file_name = f'plot_bias_vector_{epoch}.png'
                file_path = os.path.join(plot_path, file_name)
                print(f'Saving plot to {file_path}')
                plt.savefig(file_path)
                plt.show()
                plt.close()

                # Save bias_vector_np to txt
                file_name = f'bias_vector_{epoch}_{title}.txt'
                file_path = os.path.join(plot_path, file_name)
                print(f'Saving bias vector to {file_path}')
                np.savetxt(file_path, bias_vector.numpy(), fmt='%f')
            else:
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


class ExecutionEnvironment(object):
    def __init__(self, dc, gt_langs, config, graph_path, wandb_original_run_name, model_path=None, dataset_path=None):
        self.dc = dc
        self.wandb_original_run_name = wandb_original_run_name

        self.gt_langs = gt_langs
        self.target_langs = None
        self.graph_path = graph_path
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = f1_loss_function

        self.model = None
        self.optimizer = None
        # self.train_loader = None
        self.df_additional_words = None
        self.model_path = model_path if model_path is not None else f'data/3_models/{self.wandb_original_run_name}.cpickle'
        self.dataset_path = dataset_path if dataset_path is not None else f'data/9_datasets/{self.wandb_original_run_name}.cpickle'

        # saved graph
        self.G = None
        self.empty_questions_by_lang = None
        self.num_removed_gt_qid_links = 0

        # saved dataset
        self.data = None
        self.qid_node_idx_by_name = None
        self.word_node_idx_by_name = None
        self.qid_node_names = None
        self.word_node_names = None
        self.word_node_idxs_by_lang = None
        self.word_node_names_by_qid_name = None
        self.forbidden_neg_train_edges = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        # self.color_masks = None

    def set_target_langs(self, target_langs):
        self.target_langs = target_langs
        self.all_langs = set(self.gt_langs | target_langs)
        assert len(self.all_langs) == len(self.gt_langs) + len(target_langs)  # disjunct sets

    def load_graph(self):
        print(f'Loading graph from {self.graph_path}')
        with open(self.graph_path, 'rb') as f:
            gc.disable()  # disable garbage collection to speed up loading
            self.G, self.empty_questions_by_lang, self.num_removed_gt_qid_links = cPickle.load(f)
            gc.enable()
        loaded_languages = set([self.G.nodes[node]['lang'] for node in self.G.nodes])
        loaded_languages.remove('semantic_domain_question')
        self.set_target_langs(loaded_languages - self.gt_langs)
        print('Done loading graph')

    def build_network(self):
        self.G = nx.Graph()
        self.dc._load_state()
        self.set_target_langs(set(self.dc.target_langs) - self.gt_langs)

        long_qid_name_by_qid = {}  # e.g., '1.1.1.1 1' -> 'qid: 1.1.1.1 1 Moon'
        # empty_questions = {
        #     '1.5.4 3': 'Does # in ## refer to a species of moss?',
        #     '1.5.4 5': 'Does # in ## refer to a species of liverworts are there?',
        #     '3.4.1.4.5 8': "Does # in ## refer to what people say when they don't care?",
        #     '3.5.1.7.1 5': 'Does # in ## refer to what people say to thank someone?',
        #     '3.5.1.7.1 6': 'Does # in ## refer to what people say when someone thanks them?',
        #     '4.9.5.3 1': 'Does # in ## refer to relating to God or to the spirits?',
        #     '5.7.1 6': 'Does # in ## refer to what you say when you are beginning to go to sleep?',
        #     '5.7.1 10': 'Does # in ## refer to what someone says to someone else who is going to sleep?',
        #     '6.6.4.4 1': 'Does # in ## refer to working?',
        #     '8.4.5.1 10': 'Does # in ## refer to a noun phrase with no overt marker?',
        # }
        qids = list(self.dc.question_by_qid_by_lang['eng'].keys())
        # qids.extend(list(empty_questions.keys()))
        qids = [qid for qid in qids if not qid.startswith('9')]  # remove grammar SDs
        assert len(qids) == 7425

        for qid in tqdm(qids, total=len(qids), desc='Building question names'):
            cid = qid.split(' ')[0]
            question_idx = int(qid.split(' ')[1])
            sd_name = self.dc.sds_by_lang['eng'][self.dc.sds_by_lang['eng']['cid'] == cid]['category'].iloc[0]
            # try:
            question = self.dc.sds_by_lang['eng'][(self.dc.sds_by_lang['eng']['cid'] == cid) & (
                    self.dc.sds_by_lang['eng']['question_index'] == question_idx)]['question'].iloc[0]
            # except IndexError:
            #     question = empty_questions[qid]
            long_qid_name = f'qid: {sd_name} {qid} ({question})'
            long_qid_name_by_qid[qid] = long_qid_name

        # add nodes for qids
        for qid in qids:
            self.G.add_node(long_qid_name_by_qid[qid], lang='semantic_domain_question')

        # add nodes for words and edges to semantic domains and alignments
        nodes_by_lang = defaultdict(set)
        edges = set()
        for lang in tqdm(self.all_langs,
                         desc='Building network',
                         total=len(self.all_langs)):
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
                    if translation_lang == lang or translation_lang not in self.all_langs | {
                        'semantic_domain_question'} or \
                            lang not in self.gt_langs and translation_lang not in self.gt_langs:
                        # skip self-loops, ignored langs, and edges between target langs (e.g., ibo-gej)
                        continue
                    nodes_by_lang[translation_lang].add(str(translation))

                    # # skip words that only belong to qids that are not in the graph
                    # if not any(long_qid_name_by_qid[qid] in self.G.nodes for qid in translation.qids):
                    #     continue

                    # skip rare edges for higher precision and speed
                    if alignment_count < self.config['min_alignment_count']:
                        continue

                    edges.add((str(word), str(translation), alignment_count))

        # # add eng words first (for debugging)
        # self.G.add_nodes_from(nodes_by_lang['eng'], lang='eng')
        for lang in self.gt_langs:
            # add nodes for gt langs first
            self.G.add_nodes_from(nodes_by_lang[lang], lang=lang)

        for lang in nodes_by_lang.keys() - self.gt_langs:
            # add remaining nodes
            assert lang in self.target_langs, f'"{lang}" is neither a ground-truth lang nor a target lang.'
            self.G.add_nodes_from(nodes_by_lang[lang], lang=lang)
        self.G.add_weighted_edges_from(edges)

        # assert that there are word-qid edges
        assert len([edge for edge in self.G.edges if
                    'semantic_domain_question' in (self.G.nodes[edge[0]]['lang'], self.G.nodes[edge[1]]['lang'])])

        # # find the word node that the most edges to semantic_domain_questions
        # most_edges = 0
        # most_edges_node = None
        # for node in self.G.nodes:
        #     if self.G.nodes[node]['lang'] in self.gt_langs:
        #         num_edges = len([n for n in self.G.neighbors(node) if
        #                          self.G.nodes[n]['lang'] == 'semantic_domain_question'])
        #         if num_edges > most_edges:
        #             most_edges = num_edges
        #             most_edges_node = node

        # # count the words that have no edges to semantic_domain_questions
        # num_words_without_edges = 0
        # for node in self.G.nodes:
        #     if self.G.nodes[node]['lang'] in self.gt_langs:
        #         num_edges = len([n for n in self.G.neighbors(node) if
        #                          self.G.nodes[n]['lang'] == 'semantic_domain_question'])
        #         if num_edges == 0:
        #             num_words_without_edges += 1

        # # count the semantic_domain_questions that have no edges to words
        # num_sds_without_edges = 0
        # for node in self.G.nodes:
        #     if self.G.nodes[node]['lang'] == 'semantic_domain_question':
        #         num_edges = len([n for n in self.G.neighbors(node) if
        #                          self.G.nodes[n]['lang'] in self.gt_langs])
        #         if num_edges == 0:
        #             num_sds_without_edges += 1
        #
        # # count the average number of edges to words per semantic_domain_question
        # # also compute the standard deviation
        # num_edges_to_words = 0
        # num_sds = 0
        # num_edges_to_words_list = []
        # for node in self.G.nodes:
        #     if self.G.nodes[node]['lang'] == 'semantic_domain_question':
        #         num_edges = len([n for n in self.G.neighbors(node) if
        #                          self.G.nodes[n]['lang'] in self.gt_langs])
        #         num_edges_to_words += num_edges
        #         num_sds += 1
        #         num_edges_to_words_list.append(num_edges)
        # avg_num_edges_to_words = num_edges_to_words / num_sds
        # std_num_edges_to_words = np.std(num_edges_to_words_list)
        #
        # # count the average number of edges to semantic_domain_questions per word
        # # also compute the standard deviation
        # num_edges_to_sds = 0
        # num_words = 0
        # num_edges_to_sds_list = []
        # for node in self.G.nodes:
        #     if self.G.nodes[node]['lang'] in self.gt_langs:
        #         num_edges = len([n for n in self.G.neighbors(node) if
        #                          self.G.nodes[n]['lang'] == 'semantic_domain_question'])
        #         num_edges_to_sds += num_edges
        #         num_words += 1
        #         num_edges_to_sds_list.append(num_edges)
        # avg_num_edges_to_sds = num_edges_to_sds / num_words
        # std_num_edges_to_sds = np.std(num_edges_to_sds_list)
        #
        # # get the number of words
        # num_words = len([node for node in self.G.nodes if self.G.nodes[node]['lang'] in self.gt_langs])

        # assert that self.G has no node in ignored languages
        for node in self.G.nodes:
            assert self.G.nodes[node]['lang'] in self.all_langs | {'semantic_domain_question'}

        self.normalize_edge_weights()
        self.filter_edges_by_weight()
        self.find_empty_questions_by_lang([n for n in long_qid_name_by_qid.values() if n in self.G])

        self.num_removed_gt_qid_links = 0  # reset again in case we accidentally initialized it with a higher value
        # remove nodes with no edge to a GT lang node because we cannot predict their semantic domain
        for lang in tqdm(self.all_langs, total=len(self.all_langs), desc='Removing nodes with no edge to a GT lang'):
            for node in nodes_by_lang[lang]:
                if len([n for n in self.G.neighbors(node) if self.G.nodes[n]['lang'] in self.gt_langs]) == 0:
                    self.num_removed_gt_qid_links += len(
                        [n for n in self.G.neighbors(node) if self.G.nodes[n]['lang'] == 'semantic_domain_question'])
                    self.G.remove_node(node)

        print(f'Number of removed nodes with GT qid link: {self.num_removed_gt_qid_links}')

        print(f'Saving graph to {self.graph_path}')
        with open(self.graph_path, 'wb') as f:
            cPickle.dump((self.G, self.empty_questions_by_lang, self.num_removed_gt_qid_links), f)
        print('Done saving graph')

    def sort_empty_questions_by_lang(self):
        # sort by qid
        for lang in tqdm(self.all_langs, total=len(self.all_langs), desc='Sorting empty questions by lang'):
            qid_by_question = {question: self.get_qid_from_node_name(question) for question in
                               self.empty_questions_by_lang[lang]}
            self.empty_questions_by_lang[lang] = sorted(self.empty_questions_by_lang[lang],
                                                        key=lambda x: qid_by_question[x])

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
        # (This does not compromise the recall metric because does not remove word-SDQ edges.)
        removed_edges = []
        for edge in tqdm(self.G.edges(data='weight'), desc='Filtering edges by weight', total=len(self.G.edges)):
            if edge[2] < self.config['min_edge_weight']:
                removed_edges.append(edge)
        for edge in removed_edges:
            self.G.remove_edge(edge[0], edge[1])

        # assert that there are still word-word edges left
        assert len([edge for edge in self.G.edges if
                    'semantic_domain_question' not in (self.G.nodes[edge[0]]['lang'], self.G.nodes[edge[1]]['lang'])])

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
        self.sort_empty_questions_by_lang()

    def convert_empty_questions_by_lang_to_tensors(self):
        for lang in self.all_langs:
            temp = torch.zeros(len(self.qid_node_idx_by_name), dtype=torch.bool)
            for long_qid_name in self.empty_questions_by_lang[lang]:
                qid_idx = self.qid_node_idx_by_name[long_qid_name]
                temp[qid_idx] = True
            self.empty_questions_by_lang[lang] = temp

    def plot_subgraph(self, graph, node, target_node=None):
        assert target_node is None or target_node in graph.nodes
        non_latin_languages = {'npi', 'hin', 'mal', 'cmn', 'ben', 'mkn'}  # difficult to plot
        other_removed_languages = {'deu', 'hmo', 'azb', 'tpi', 'yor'}  # removed to improve readability
        # order target_langs alphabetically
        target_langs = ['semantic_domain_question'] + sorted(
            list(self.all_langs - non_latin_languages - other_removed_languages))
        palette = sns.color_palette('pastel')  # 'tab20'
        palette = {lang: color for lang, color in zip(target_langs, palette)}
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
            for neighbor_2nd_order in filtered_word_graph.neighbors(neighbor_1st_order):
                if filtered_word_graph.nodes[neighbor_2nd_order][
                    'lang'] == 'semantic_domain_question' or neighbor_2nd_order == target_node:
                    neighbors_2nd_order.add(neighbor_2nd_order)
                for neighbor_3rd_order in filtered_word_graph.neighbors(neighbor_2nd_order):
                    if neighbor_3rd_order == target_node:
                        neighbors_2nd_order.add(neighbor_2nd_order)
                        neighbors_3rd_order.add(neighbor_3rd_order)

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

        # remove all langs from palette and target_langs that are not in the subgraph
        palette = {lang: color for lang, color in palette.items() if
                   lang in {lang for node, lang in displayed_subgraph.nodes(data='lang')}}
        target_langs = [lang for lang in target_langs if
                        lang in {lang for node, lang in displayed_subgraph.nodes(data='lang')}]

        # use a different node color for each language
        node_colors = [palette[data['lang']] for node_name, data in displayed_subgraph.nodes(data=True)]

        # rename 'semantic_domain_question' to 'SDQ'
        if 'semantic_domain_question' in palette:
            palette['SDQ'] = palette.pop('semantic_domain_question')
            target_langs[0] = 'SDQ'

        # put SDQ first in the palette
        palette = {'SDQ': palette.pop('SDQ'), **palette}

        # show all the colors in a legend
        plt.legend(handles=[Patch(color=palette[lang], label=lang) for lang in target_langs], prop={'size': 20})

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(displayed_subgraph)

        # draw nodes
        nx.draw_networkx_nodes(displayed_subgraph, pos=pos, node_color=node_colors)

        def get_node_label_name(node_name):
            lang, text = node_name.split(': ')
            if lang != 'qid':
                return text
            else:
                qid = self.get_qid_from_node_name(node_name).replace(' ', '-')
                sd = self.get_sd_from_node_name(node_name)
                # text = '\n'.join(textwrap.wrap(text, width=20))
                return f'{qid}\n{sd}'

        # draw only word texts as node labels
        nx.draw_networkx_labels(displayed_subgraph, pos=pos,
                                labels={node_name: get_node_label_name(node_name)
                                        for node_name in displayed_subgraph.nodes()}, font_size=20)

        # draw edges (thicker edges for more frequent alignments)
        for edge in displayed_subgraph.edges(data='weight'):
            weight = edge[2]

            if 'semantic_domain_question' == graph.nodes[edge[0]]['lang']:
                color, _, _, _ = self.get_edge_color(edge[1], edge[0])
                weight = 0.2
            elif 'semantic_domain_question' == graph.nodes[edge[1]]['lang']:
                color, _, _, _ = self.get_edge_color(edge[0], edge[1])
                weight = 0.2
            else:
                color = 'gray'
            if color in ('red', 'orange'):
                continue

            if color == 'green':
                color = palette['SDQ']

            nx.draw_networkx_edges(displayed_subgraph, pos=pos, edgelist=[edge],
                                   # caution: might fail in debug mode with Python 3.10 instead of Python 3.9 or 3.11
                                   width=[math.log(weight * 10) + 1], edge_color=color)  # , alpha=0.5)

        # draw edge labels with 2 decimal places
        edge_weights = nx.get_edge_attributes(displayed_subgraph, 'weight')
        edge_labels = dict([((u, v), f"{d['weight']:.2f}") for u, v, d in displayed_subgraph.edges(data=True)])
        # skip edges to semantic domains
        edge_labels = {k: v for k, v in edge_labels.items() if
                       graph.nodes[k[0]]['lang'] != 'semantic_domain_question' and
                       graph.nodes[k[1]]['lang'] != 'semantic_domain_question'}
        if len(edge_weights):
            nx.draw_networkx_edge_labels(displayed_subgraph, pos, edge_labels=edge_labels, font_size=15)

        lang, text = node.split(': ')
        plt.title(f'Words aligned with "{text}" ({lang}) and their linked SDQs', fontsize=20)
        plt.tight_layout()
        plt.margins(x=0.1, y=0.1)
        plt.savefig(f'plots/subgraph_{node}.pdf')
        plt.show()

    def train(self):
        ys = []
        y_hats = []
        # node_idxs = []
        train_losses = []
        self.model.train()

        train_loader = NeighborLoader(
            self.data,
            num_neighbors=[-1],
            # [-1] vs [-1, -1] makes a difference if normalization=1 because it changes the neighbor's edge weights
            batch_size=self.config['batch_size'],
            input_nodes=self.data.train_mask.to(self.device),  # & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        )  # for color_mask in self.color_masks if
        # (self.data.train_mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        # for train_loader in self.train_loaders:
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch.to(self.device)
            y = batch.y[:batch.batch_size]
            # x = batch.x.clone()
            # x = x.to_dense()
            # x[:batch.batch_size] = 0  # mask out the nodes to be predicted
            y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
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
            val_loss, _, _, _, precision, recall, f1, val_soft_precision, val_soft_recall = self.evaluate(
                self.data.val_mask, threshold=0.999, compute_ideal_threshold=False, plots=plots, num_frame=epoch)

            train_loss = self.train()
            soft_precision_by_lang, soft_recall_by_lang, f1_loss_by_lang = self.evaluate_each_gt_lang(
                self.data.val_mask, threshold=0.999, use_soft_metrics=True)

            log_dict = {"train loss": train_loss, "val loss": val_loss,
                        "val soft precision": val_soft_precision, "val soft recall": val_soft_recall}
            for lang in self.gt_langs:
                log_dict.update({f"val precision {lang}": soft_precision_by_lang[lang],
                                 f"val recall {lang}": soft_recall_by_lang[lang],
                                 f"val f1 loss {lang}": f1_loss_by_lang[lang]})
            wandb.log(log_dict)
            wandb.watch(self.model)

            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Soft Precision: {val_soft_precision:.3f}, Val Soft Recall: {val_soft_recall:.3f}")

            checkpoint_path = f"data/3_models/model_{wandb.run.name}_epoch{epoch}_val-loss{val_loss:.3f}_checkpoint.bin"
            if early_stopper.early_stop(val_loss, epoch, self.save_model, checkpoint_path):
                print("Early stopping")
                self.create_gif()  # outcomment for DNAConv
                self.load_model(early_stopper.last_checkpoint)
                break

    def compute_loss_with_empty_questions(self, y_hat, y, node_idxs, function=None):
        # do not count candidates at unknown positions
        node_langs = [self.word_node_names[i.item()].split(': ')[0] for i in node_idxs]
        empty_questions = [self.empty_questions_by_lang[lang] for lang in node_langs]
        empty_questions = torch.stack(empty_questions).to(self.device)
        if function is None:
            return self.criterion(y_hat, y, empty_questions)
            # return self.criterion(y_hat.to(self.device).float(), y.to(self.device).float())#, empty_questions)
        else:
            return function(y_hat, y, empty_questions)

    def compute_metric_tensor_with_empty_questions(self, candidates, node_idxs):
        # do not count candidates at unknown positions
        node_langs = [self.word_node_names[i.item()].split(': ')[0] for i in node_idxs]
        empty_questions = [self.empty_questions_by_lang[lang] for lang in node_langs]
        valid_questions = torch.logical_not(torch.stack(empty_questions)).to(self.device)
        candidates = candidates.to(self.device)
        return torch.logical_and(candidates, valid_questions)

    def compute_metric_with_empty_questions(self, candidates, node_idxs):
        result_tensor = self.compute_metric_tensor_with_empty_questions(candidates, node_idxs)
        result_num = torch.sum(result_tensor).item()
        return result_num, result_tensor

    def get_node_names_and_label_from_edge_label_index(self, edge_label, edge_label_index, idx):
        label = edge_label[idx].item()
        sd_idx = edge_label_index[(0, idx)].item()
        word_idx = edge_label_index[(1, idx)].item()
        qid = self.qid_node_names[sd_idx]
        word = self.word_node_names[word_idx]
        return qid, word, word_idx, label

    def find_similar_words_in_sd(self, word, sd_name):
        # use word_node_names_by_qid_name and normalized edit distance to find similar words
        # return a list of similar words
        similar_words = []
        word_plain = word.split(': ')[1]
        for w in self.word_node_names_by_qid_name[sd_name]:
            w_plain = w.split(': ')[1]
            dist = editdistance.eval(word_plain, w_plain) / len(word_plain)
            if dist <= 0.4:
                similar_words.append((w, dist))
        similar_words.sort(key=lambda x: x[1])
        return similar_words

    def export_dictionary_to_html(self, lang, path, predicted_words_by_qid):
        # <h2> [SD Name] </h2>
        # (1) [QID]
        # - [Word 1], [Word 2], [Word 3], ...
        # (2) [QID]
        # - [Word 1], [Word 2], [Word 3], ...
        # ...
        # <h2> [SD Name] </h2>
        # ...

        print(f'Exporting dictionary to {path}...')

        # cluster all qids by their SD name (e.g., qid: '1.1.1.1 1 Moon' -> sd: '1.1.1.1 Moon')
        qids_by_sd_name = defaultdict(list)
        for qid in self.qid_node_names:
            sd_name = self.get_sd_from_node_name(qid)
            sd_id = self.get_qid_from_node_name(qid).split(' ')[0]
            sd_name = sd_id + ' ' + sd_name
            qids_by_sd_name[sd_name].append(qid)

        # sort by sd name
        qids_by_sd_name = dict(sorted(qids_by_sd_name.items(), key=lambda x: x[0]))

        self.dc._load_state()
        self.set_target_langs(set(self.dc.target_langs) - self.gt_langs)
        sds_df = self.dc.sds_by_lang[lang]

        # write to html file
        with open(path, 'w') as f:
            for sd_name, qids in qids_by_sd_name.items():
                f.write(f'<h2>{sd_name}</h2>\n')
                for qid_node in qids:
                    question = self.get_question_from_node_name(qid_node)
                    cid, q_num = self.get_qid_from_node_name(qid_node).split(' ')
                    try:
                        row = sds_df[(sds_df['cid'] == cid) & (sds_df['question_index'] == int(q_num))].iloc[0]
                        _, _, original_answers = self.dc.split_sd_answers(row)
                        # original_answers = \
                        # str(sds_df[(sds_df['cid'] == cid) & (sds_df['question_index'] == int(q_num))]['answer'].iloc[0])
                    except IndexError:
                        original_answers = []
                    original_answers = ', '.join(original_answers)
                    f.write(f'<p>({q_num}) {question}</p>\n')
                    f.write(f' <p><i>• {original_answers}')
                    if len(original_answers) > 0 and original_answers[-1] != ',':
                        f.write(',')

                    for word_node in predicted_words_by_qid[qid_node]:
                        word_lang, word = word_node.split(': ')
                        if word_lang == lang:
                            color, _, translation, _ = self.get_edge_color(word_node, qid_node)
                            if color == 'green':
                                continue
                            elif color in ('orange', 'gray'):
                                new_color = 'green'
                            else:
                                new_color = color
                            f.write(f' <span style="color:{new_color}"><b>{word}</b></span>')
                            if translation not in ('', '#N/A'):
                                f.write(f'<span style="color:blue">{translation}</span>')
                            f.write(',')
                    f.write('</i></p>\n')
                f.write('<br>\n')

        print('Done exporting dictionary for ' + lang)

    def add_prediction_row_to_csv(self, file, prediction):
        semantic_domain = self.get_sd_from_node_name(prediction[0])
        qid = self.get_qid_from_node_name(prediction[0])
        question = self.get_question_from_node_name(prediction[0])
        lang, word = prediction[1].split(': ')
        key = f'{qid}#{lang}#{word}'

        color = self.get_edge_color(prediction[1], prediction[0])[0]
        if color == 'red':
            previous_answer = 0
        elif color == 'orange':
            previous_answer = 1
        elif color == 'green':
            previous_answer = 'ground-truth'
        else:
            previous_answer = 'n/a'

        line = f'{key},{qid},"{semantic_domain}",{lang},{word},"{question}",{prediction[2]},{prediction[3]:.2f},{prediction[4]},{previous_answer}\n'
        # print(line)
        file.write(line)
        # file.write(key + ',')  # key
        # file.write(qid + ',')  # qid
        # file.write('"' + semantic_domain + '",')  # semantic domain
        # file.write(lang + ',')  # language
        # file.write(word + ',')  # word
        # file.write('"' + question + '",')  # question
        # file.write(str(prediction[2]) + ',')  # score
        # file.write(f'{prediction[3]:.2f},')  # raw score
        # file.write(str(prediction[4]) + ',')  # is correct
        # file.write(str(previous_answer) + '\n')  # previous_answer
        # file.write('"' + str(prediction[5]) + '"\n')  # similar words
        # file.write('"' + str(prediction[6]) + '"\n')  # explanation

    def print_human_readable_predictions(self, y, y_hat, link_probs, mask, threshold, max_print_words=10,
                                         save_to_csv_path=None, word_idx_map=None, max_print_total=100):
        if save_to_csv_path:
            print(f'Saving predictions to {save_to_csv_path}...')
            file = open(save_to_csv_path, 'w')
            file.write('key,qid,semantic_domain,lang,word,question,score,raw_score,is_correct,previous_answer\n')
        else:
            print("Printing human readable predictions...")
            file = None

        # for each element in out, lookup all edges in eval_data[self.TARGET_EDGE_TYPE].edge_label_index
        count_printed_words = 0
        count_printed_total = 0
        predicted_words_by_qid_node = defaultdict(list)
        assert (mask is None or len(mask) == len(link_probs))
        for word_idx, qid_preds in tqdm(enumerate(link_probs), total=len(link_probs)):
            if mask is not None and not mask[word_idx]:
                continue
            printed = False

            # find all qid_idxs for positive predictions (3x faster than manual filtering)
            qid_idxs = torch.nonzero(qid_preds >= threshold).flatten()

            for qid_idx in qid_idxs:
                prob = qid_preds[qid_idx]
                pred = 1  # if prob >= threshold else 0
                # if pred == 0:  # and is_correct:
                #     continue
                label = y[word_idx][qid_idx] if y is not None else None
                is_correct = label is None or (pred == label).item()

                original_word_idx = word_idx_map[word_idx].item() if word_idx_map is not None else word_idx
                word = self.word_node_names[original_word_idx]
                qid = self.qid_node_names[qid_idx]
                pred_raw = y_hat[word_idx][qid_idx]
                assert not is_correct or word in self.word_node_names_by_qid_name[qid]

                prefix = '! ' if not is_correct and label is not None else '  '  # highlight false positives and false negatives
                similar_words = []
                explanation = ''
                if pred == 1 and not is_correct:
                    similar_words = self.find_similar_words_in_sd(word, qid)
                    explanation = self.explain_false_positive(original_word_idx, qid_idx)

                predicted_words_by_qid_node[qid].append(word)
                if save_to_csv_path:
                    if pred == 1:
                        self.add_prediction_row_to_csv(file, [qid, word, prob.item(), pred_raw.item(), is_correct])
                else:
                    print(
                        f'{prefix} prediction: {pred} ({pred_raw:.0f} --> {prob:.2f}/{threshold:.2f}), actual: {label}, {word} ({word_idx}) <---> {qid} ({qid_idx})',
                        similar_words, explanation)
                printed = True
                count_printed_total += 1
            count_printed_words += printed
            if count_printed_words >= max_print_words or count_printed_total >= max_print_total:
                print(f'... and {len(link_probs) - max_print_words} more words')
                break
        print(f'Printed {count_printed_words} words and {count_printed_total} total predictions')

        if file:
            file.write('Done\n')
            file.close()

        return predicted_words_by_qid_node

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
                      0.999]  # np.linspace(0.00, 1.00, int((1.00 - 0.00) / 0.01) + 1)
        func = partial(compute_f1_score_with_threshold, target, predicted)
        # optional todo: try to use Pool to parallelize
        f1_scores = tqdm(map(func, thresholds),
                         desc='Finding ideal threshold to maximize F1 score',
                         total=len(thresholds))
        thresholds_and_f1_scores = list(zip(thresholds, f1_scores))
        thresholds_and_f1_scores.sort(key=lambda x: x[1], reverse=True)
        print('Top 3 thresholds and their F1 scores:', thresholds_and_f1_scores[:3])
        return thresholds_and_f1_scores[0][0]

    @staticmethod
    def find_ideal_threshold_to_maximize_recall_for_given_precision(target, predicted, target_precision):
        thresholds = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 0.99, 0.999]
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
            print(
                'WARNING: This evaluation is less expressive because it also includes the training set. It might also crash because it requires more memory.')
            mask = torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.device)
        self.model.eval()
        y_hats = []
        ys = []
        node_idxs = []

        eval_loader = NeighborLoader(
            self.data,
            num_neighbors=[-1],
            batch_size=self.config['batch_size'],
            input_nodes=mask.to(self.device),  # & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        )  # for color_mask in self.color_masks if (mask.to(self.device) & color_mask.to(self.device)).sum() > 0

        # for eval_loader in tqdm(eval_loaders, desc='Evaluating...', total=len(eval_loaders)):
        for batch in eval_loader:
            # x = batch.x.clone()
            # x = x.to_dense()
            # x[:batch.batch_size] = 0  # mask out the nodes to be predicted
            y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)[
                    :batch.batch_size]
            y_hats.append(y_hat.detach().cpu())
            y = batch.y[:batch.batch_size]
            ys.append(y.detach().cpu())
            node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())

        y_hat = torch.cat(y_hats, dim=0)
        y = torch.cat(ys, dim=0)
        node_idxs = torch.cat(node_idxs, dim=0)

        # assert y.sum() > 0  # at least one positive example
        if y.sum() == 0:
            print('WARNING: No positive examples in evaluation set')
        assert (len(y) == len(y_hat))
        assert (len(y) == len(node_idxs))
        assert (len(y) == mask.sum())
        print(f'Computing evaluation metrics for {len(y)} words...')
        eval_loss = self.compute_loss_with_empty_questions(y_hat, y, node_idxs)
        soft_precision = self.compute_loss_with_empty_questions(y_hat, y, node_idxs,
                                                                soft_precision_function)
        soft_recall = self.compute_loss_with_empty_questions(y_hat, y, node_idxs,
                                                             soft_recall_function)
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

        false_positives = false_negatives = precision = recall = f1 = -1.0

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
            false_positives, _ = self.compute_metric_with_empty_questions(
                candidates, node_idxs)
            print(
                f'False positives: {false_positives} ({false_positives / num_predictions * 100:.2f} %), raw: {raw_false_positives}')

            candidates = torch.logical_and(link_probs_tensor < threshold, y_tensor == 1)
            raw_false_negatives = torch.sum(candidates).item()
            false_negatives, _ = self.compute_metric_with_empty_questions(candidates, node_idxs)
            false_negatives += self.num_removed_gt_qid_links * (mask.sum().item() / len(mask))
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
            print(f'Precision: {precision:.2f}')  # , raw: {raw_precision:.2f}')

            raw_recall = raw_true_positives / (
                    raw_true_positives + raw_false_negatives) if raw_true_positives + raw_false_negatives > 0 else 0.0
            recall = true_positives / (
                    true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
            print(f'Recall: {recall:.2f}')  # , raw: {raw_recall:.2f}')

            raw_f1 = 2 * (raw_precision * raw_recall) / (
                    raw_precision + raw_recall) if raw_precision + raw_recall > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            print(f'F1: {f1:.2f}')  # , raw: {raw_f1:.2f}')

        if 'explain false positives' in plots:
            candidates = torch.logical_and(link_probs_tensor >= threshold, y_tensor == 0)
            false_positives_tensor = self.compute_metric_tensor_with_empty_questions(candidates, node_idxs)
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
            self.model.visualize_weights(self.wandb_original_run_name, num_frame, eval_loss, self.config['plot_path'])

        if print_highest_qid_correlations:
            self.model.print_highest_qid_correlations(self.qid_node_names)

        if print_human_readable_predictions:
            self.print_human_readable_predictions(y, y_hat, link_probs, None, threshold, word_idx_map=node_idxs)

        return eval_loss, ideal_threshold, false_positives, false_negatives, precision, recall, f1, soft_precision, soft_recall

    @staticmethod
    def convert_edge_index_to_set_of_tuples(edge_index):
        return {(u, v) for u, v in edge_index.transpose(0, 1).tolist()}

    def save_model(self, path=None):
        path = path if path is not None else self.model_path
        print(f'Saving model to {path}')
        torch.save(self.model.state_dict(), path)

    def update_original_run_name(self, model_path):
        # extract original name from path (e.g. 'model_fanciful-gorge-55_epoch417_val-loss0.917_checkpoint.bin' -> 'fanciful-gorge-55')
        try:
            self.wandb_original_run_name = re.search(r'(?<=model_)(.*?)(?=[_\.])', os.path.basename(model_path)).group(
                1)
        except AttributeError:
            self.wandb_original_run_name = os.path.basename(model_path)

    def load_model(self, path=None):
        path = path if path is not None else self.model_path
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
        ## clone the data to avoid changing the original data
        pred_data = self.data  # .clone()
        pred_data.to(self.device)

        # build mask for test_word
        test_word_idx = self.word_node_idx_by_name[test_word]
        mask = torch.zeros(len(pred_data.x), dtype=torch.bool)
        mask[test_word_idx] = True
        mask = mask.to(self.device)

        # mask out the current word
        # pred_data.x = pred_data.x.to_dense()
        # pred_data.x[mask] = 0

        y_hat = self.model(pred_data.x, pred_data.edge_index, pred_data.edge_weight)
        link_probs = y_hat.sigmoid()

        # print all positive predictions
        self.print_human_readable_predictions(None, y_hat, link_probs, mask, threshold)
        print('\n')

    @torch.no_grad()
    def predict_for_languages(self, langs, threshold):
        # clone the data to avoid changing the original data
        pred_data = self.data  # .clone()
        pred_data.to(self.device)

        # build mask for languages
        mask = torch.zeros(len(pred_data.x), dtype=torch.bool)
        for lang in langs:
            mask[self.word_node_idxs_by_lang[lang]] = True
        mask = mask.to(self.device)

        node_loader = NeighborLoader(
            pred_data,
            num_neighbors=[-1],
            batch_size=self.config['batch_size'],
            input_nodes=mask.to(self.device),  # & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        )  # for color_mask in self.color_masks if (mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        ys = []
        y_hats = []
        node_idxs = []
        # for node_loader in node_loaders:
        for batch in node_loader:
            batch.to(self.device)
            y = batch.y[:batch.batch_size]
            ys.append(y.detach().cpu())
            # x = batch.x.clone()
            # x = x.to_dense()
            # x[:batch.batch_size] = 0  # mask out the nodes to be predicted
            y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)
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
        predicted_words_by_qid_node = self.print_human_readable_predictions(y, y_hat, link_probs,
                                                                            positive_prediction_mask, threshold,
                                                                            max_print_words=int(1e10),
                                                                            save_to_csv_path=f'data/6_results/predicted_sds_{langs}_{self.wandb_original_run_name}.csv',
                                                                            word_idx_map=node_idxs,
                                                                            max_print_total=int(1e10))
        print('\n')
        return predicted_words_by_qid_node

    def explain_false_positive(self, node_idx, qid_idx, fp_node_name_by_qid_by_node=None):
        node_name = self.word_node_names[node_idx]
        qid_name = self.qid_node_names[qid_idx]
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
        # x = batch.x.clone()
        # x = x.to_dense()
        # x[:batch.batch_size] = 0  # mask out the node to be predicted
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
            x_neighbor[neighbor_idx] = 0  # mask out the neighbor
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

        qid = self.get_qid_from_node_name(qid_name)
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
                # assert node_name not in self.word_node_names_by_qid_name[qid_name]
                if node_name in self.word_node_names_by_qid_name[qid_name]:
                    print(
                        f'WARNING: {node_name} is marked as a false positive, but it is in the GT dict for {qid_name}.')
                word_has_gpt4_answer = True

        if not word_has_gpt4_answer:
            if node_name in self.word_node_names_by_qid_name[qid_name]:
                color = 'green'  # the GT dict says the word belongs to the QID
                is_manually_verified = True
            else:
                color = 'gray'  # unknown word

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
        # sort by qid
        qid_by_question_node_name = {k: self.get_qid_from_node_name(k) for k in fp_node_names_by_qid_by_node.keys()}
        fp_node_names_by_qid_by_node = {k: v for k, v in sorted(fp_node_names_by_qid_by_node.items(),
                                                                key=lambda item: qid_by_question_node_name[item[0]])}

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
                    const colors = ['lightgray', 'lightgreen', 'pink', 'yellow'];
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

                            // Print the group type and the color of the first button (if it is not gray)
                            color = button.style.backgroundColor;
                            if (color != 'lightgray' && color != '#f0f0f0') {
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

    def evaluate_each_gt_lang(self, mask, threshold, use_soft_metrics=False):
        precision_by_lang = {}
        recall_by_lang = {}
        f1_by_lang = {}
        for lang in self.gt_langs:
            print(f'\nPerformance for {lang}:')
            lang_mask = torch.zeros(len(self.data.x), dtype=torch.bool)
            lang_mask[self.word_node_idxs_by_lang[lang]] = True
            lang_mask = lang_mask.to(self.device)
            lang_mask &= mask.to(self.device)
            eval_loss, _, _, _, precision, recall, f1, soft_precision, soft_recall = self.evaluate(
                lang_mask, [], False, True, threshold,
                compute_ideal_threshold=False)
            if use_soft_metrics:
                precision_by_lang[lang] = soft_precision
                recall_by_lang[lang] = soft_recall
                f1_by_lang[lang] = 1.0 - eval_loss
            else:
                precision_by_lang[lang] = precision
                recall_by_lang[lang] = recall
                f1_by_lang[lang] = f1
        return precision_by_lang, recall_by_lang, f1_by_lang

    def test(self, print_human_readable_predictions=False, threshold=None,
             compute_ideal_threshold=True, plots=None, json_result_path=None):
        plots = [] if plots is None else plots

        print('\nValidation set performance:')
        _, ideal_threshold, _, _, _, _, _, _, _ = self.evaluate(
            self.data.val_mask, plots, print_human_readable_predictions, True, threshold,
            compute_ideal_threshold=compute_ideal_threshold, print_highest_qid_correlations=True)

        print('\nTest set performance:')
        test_loss, _, false_positives, false_negatives, test_precision, test_recall, test_f1, _, _ = self.evaluate(
            self.data.test_mask, [], print_human_readable_predictions, True,
            ideal_threshold if threshold is None else threshold, compute_ideal_threshold=False)

        if wandb.run is not None:
            wandb.log({"test loss": test_loss, "test soft F1": 1.0 - test_loss,
                       "val ideal threshold": ideal_threshold,
                       "false positives": false_positives, "false negatives": false_negatives,
                       "precision": test_precision,
                       "recall": test_recall, "F1": test_f1})
            wandb.watch(self.model)

        precision_by_lang, recall_by_lang, f1_by_lang = self.evaluate_each_gt_lang(self.data.test_mask,
                                                                                   ideal_threshold if threshold is None else threshold)

        # save the results to a csv file
        precision_by_lang = list(sorted(precision_by_lang.items(), key=lambda item: item[0]))
        recall_by_lang = list(sorted(recall_by_lang.items(), key=lambda item: item[0]))
        f1_by_lang = list(sorted(f1_by_lang.items(), key=lambda item: item[0]))
        path = f'data/6_results/precision_recall_f1.csv'
        print(f'Saving precision, recall, and F1 to {path}')
        if not os.path.exists(path):
            # create file empty file if it does not exist yet
            with open(path, 'w') as f:
                f.write('')
        with open(path, 'a') as f:
            f.write('run_name,graph_path,')
            f.write(','.join(
                [f'{precision_by_lang[idx][0]}_precision,{recall_by_lang[idx][0]}_recall,{f1_by_lang[idx][0]}_f1' for
                 idx in range(len(self.gt_langs))]) + '\n')
            f.write(f'{wandb.run.name},{self.graph_path},')
            f.write(','.join([f'{precision_by_lang[idx][1]},{recall_by_lang[idx][1]},{f1_by_lang[idx][1]}' for idx in
                              range(len(self.gt_langs))]) + '\n')

        # save the results to a json file
        if json_result_path is not None:
            print(f'Saving precision, recall, and F1 to {json_result_path}')
            results = {}
            for idx in range(len(self.gt_langs)):
                results[precision_by_lang[idx][0]] = {'precision': precision_by_lang[idx][1],
                                                      'recall': recall_by_lang[idx][1], 'f1': f1_by_lang[idx][1]}
            with open(json_result_path, 'w') as f:
                json.dump(results, f, indent=4)

        # if eval_train_set:
        #     print('\nTrain set performance:')
        #     self.evaluate(self.data.train_mask, [], print_human_readable_predictions, True,
        #                   ideal_threshold if threshold is None else threshold, compute_ideal_threshold=False)

        return test_loss, ideal_threshold, test_f1, test_precision

    def split_dataset(self):
        split = T.RandomNodeSplit(
            num_val=0.10,
            num_test=0.10,
        )
        self.data = split(self.data)
        print(self.data)

    # def partition_graph(self):
    #     # solve n-color problem for multiple hops
    #     hop_graph = self.G.copy()
    #
    #     # remove qid nodes
    #     hop_graph.remove_nodes_from(self.qid_node_names)
    #
    #     print('Partitioning graph...')
    #     color_by_node = nx.greedy_color(hop_graph)
    #     print('Done partitioning graph')
    #
    #     # create a list of bit masks (one for each color)
    #     self.color_masks = []
    #     for color in range(max(color_by_node.values()) + 1):
    #         mask = torch.zeros(hop_graph.number_of_nodes(), dtype=torch.bool)
    #         for node, node_color in color_by_node.items():
    #             if node_color == color:
    #                 node_idx = self.word_node_idx_by_name[node]
    #                 mask[node_idx] = True
    #
    #         self.color_masks.append(mask.to(self.device))
    #
    #     # assert that the masks are complete
    #     assert (sum([m.sum() for m in self.color_masks]).item() == len(self.color_masks[0]))

    def get_qid_from_node_name(self, node_name):
        try:
            return re.search(r'([0-9]+.*?)(?= \()', node_name).group(1)
        except AttributeError:
            print(f'ERROR: Could not extract QID from node name {node_name}')
            raise AttributeError

    def get_sd_from_node_name(self, node_name):
        return re.search(r'^([^0-9]+)', node_name.split(': ')[1]).group(1).strip()

    def get_question_from_node_name(self, node_name):
        return re.search(r'\((.*)\)', node_name).group(1)

    def plot_class_counts(self):
        print('Plotting class counts...')

        # count how often each class occurs in self.data
        class_counts = torch.sum(self.data.y, dim=0).cpu().numpy()
        class_counts = {f'{self.qid_node_names[i]}': count for i, count in enumerate(class_counts)}

        # get the actual QIDs
        class_counts = {self.get_qid_from_node_name(key): count for key, count in class_counts.items()}

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
        # Caution: This requires lots of storage! (~ 15GB)
        if os.path.exists(self.dataset_path):
            print(f'File {self.dataset_path} already exists, skipping saving the dataset.')
            return

        print(f'Saving dataset to {self.dataset_path}')
        with open(self.dataset_path, 'wb') as f:
            cPickle.dump((self.data, self.qid_node_idx_by_name, self.word_node_idx_by_name, self.qid_node_names,
                          self.word_node_names, self.word_node_idxs_by_lang, self.word_node_names_by_qid_name,
                          self.forbidden_neg_train_edges, self.train_data, self.val_data, self.test_data), f)
        print('Done saving dataset.')

    def load_dataset(self, path=None):
        # load the dataset with cPickle
        path = path if path is not None else self.dataset_path
        print(f'Loading dataset from {path}')
        with open(path, 'rb') as f:
            gc.disable()  # disable garbage collection to speed up loading
            self.data, self.qid_node_idx_by_name, self.word_node_idx_by_name, self.qid_node_names, self.word_node_names, \
                self.word_node_idxs_by_lang, self.word_node_names_by_qid_name, self.forbidden_neg_train_edges, \
                self.train_data, self.val_data, self.test_data = cPickle.load(f)
            self.data.to(self.device)
            gc.enable()  # enable garbage collection again
        self.convert_empty_questions_by_lang_to_tensors()
        print('Done loading dataset.')

    def build_dataset(self, additional_target_lang=None):
        # self.build_network()  # also contains languages without gt sds
        self.load_graph()
        if additional_target_lang is not None:
            assert additional_target_lang in self.target_langs

        subgraph = self.G.subgraph([n for n in self.G.nodes if
                                    self.G.nodes[n]['lang'] in self.gt_langs.union(
                                        {'semantic_domain_question',
                                         additional_target_lang} if additional_target_lang is not None else {
                                            'semantic_domain_question'})])
        self.G, complete_graph = subgraph, self.G

        self.qid_node_names = [node for node in self.G.nodes if
                               self.G.nodes[node]["lang"] == "semantic_domain_question"]
        self.word_node_names = [node for node in self.G.nodes if
                                self.G.nodes[node]["lang"] != "semantic_domain_question"]
        word_nodes_set = set(self.word_node_names)

        # Create dictionaries to map nodes to indices
        self.qid_node_idx_by_name = {node: i for i, node in enumerate(self.qid_node_names)}
        self.word_node_idx_by_name = {node: i for i, node in enumerate(self.word_node_names)}

        self.word_node_names_by_qid_name = {sd_node: list(self.G.neighbors(sd_node)) for sd_node in self.qid_node_names}

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
        print('Number of words by language:' + str(num_words_by_lang))
        print('Total number of words: ' + str(sum(num_words_by_lang.values())))

        self.convert_empty_questions_by_lang_to_tensors()

        if False and not torch.cuda.is_available():  # hack to run this only on my laptop
            self.plot_subgraph(complete_graph, 'meu: varani')  # true positive
            self.plot_subgraph(complete_graph, 'ibo: ahụọnụ')  # true positive
            self.plot_subgraph(complete_graph, 'gej: màmayɔviwoa')  # true positive

            self.plot_subgraph(complete_graph, 'eng: overdrive')  # false positive
            self.plot_subgraph(complete_graph, 'ind: enak')  # false positive

            self.plot_subgraph(complete_graph, 'swh: kishindo')  # false negative
            self.plot_subgraph(complete_graph, 'fra: lumineux')  # false negative

            self.plot_subgraph(complete_graph, 'eng: earth')
            self.plot_subgraph(complete_graph, 'eng: responsibility')
            self.plot_subgraph(complete_graph, 'eng: six')
            self.plot_subgraph(complete_graph, 'eng: left')
            self.plot_subgraph(complete_graph, 'eng: paper')
            self.plot_subgraph(complete_graph, 'eng: camel')
            self.plot_subgraph(complete_graph, 'eng: pelican')
            self.plot_subgraph(complete_graph, 'eng: blacksmith')
            self.plot_subgraph(complete_graph, 'eng: crane')
            self.plot_subgraph(complete_graph, 'eng: grandchild')
            self.plot_subgraph(self.G, 'qid: Surrender 4.8.3.4 1 (What words refer to surrendering to an enemy?)',
                               target_node='qid: Evaluate, test 3.2.2.3 1 (What words refer to the process of determining the truth of something?)')
            # self.plot_subgraph(self.G, 'eng: heavy')

            # # plot neighborhoods
            # self.plot_subgraph(self.G, random.choice(
            #     [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "ind"]))
            # self.plot_subgraph(self.G, random.choice(
            #     [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "fra"]))
            # self.plot_subgraph(complete_graph, random.choice(
            #     [node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == "deu"]))
            # self.plot_subgraph(complete_graph, random.choice(
            #     [node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == "gej"]))
            exit(0)

        # # Create language feature (for word nodes)
        # lang_feature = torch.zeros(len(self.word_node_names), len(self.all_langs))
        # for node in self.word_node_names:
        #     lang = complete_graph.nodes[node]["lang"]
        #     lang_feature[self.word_node_idx_by_name[node]][self.all_langs.index(lang)] = 1
        #     # lang_feature[self.word_node_idx_by_name[node]][0] = 1

        # Create the node degree feature
        degree_feature = torch.zeros(len(self.word_node_names), 1)
        for node in self.word_node_names:
            degree_feature[self.word_node_idx_by_name[node]] = self.G.degree(
                node)  # todo: subtract links to semantic domain questions

        # Create the weighted node degree feature
        weighted_degree_feature = torch.zeros(len(self.word_node_names), 1)
        for node in self.word_node_names:
            weighted_degree_feature[self.word_node_idx_by_name[node]] = self.G.degree(node,
                                                                                      weight='weight')  # todo: subtract links to semantic domain questions

        # Create QID feature
        indices = []
        values = []
        for node in tqdm(self.word_node_names, desc='Creating QID feature', total=len(self.word_node_names)):
            # set a 1 for each QID to which the word is connected
            for neighbor in self.G.neighbors(node):
                if neighbor in self.qid_node_names:
                    indices.append([self.word_node_idx_by_name[node], self.qid_node_idx_by_name[neighbor]])
                    values.append(1)
        qid_feature = torch.sparse_coo_tensor(torch.tensor(indices).transpose(0, 1), torch.tensor(values),
                                              size=(len(self.word_node_names), len(self.qid_node_names))).to_dense()

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

        ## normalize the pos weights by the number of positive examples in the training set
        ## we have few positive examples, so we need to weight them more
        # y = self.data.y[self.data.train_mask]
        # pos_weight = (y == 0.).sum() / y.sum()
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        # self.partition_graph()
        self.save_dataset()

    def evaluate_model(self, model_path, dataset_path, json_result_path):
        print(f'Evaluating {model_path}')

        path_1 = model_path
        self.update_original_run_name(path_1)
        # self.build_dataset()

        self.load_graph()  # only use this in combination with self.load_dataset()
        self.load_dataset(dataset_path)  # crashes for 'deu'

        # for lang in tqdm(self.gt_langs, desc='Making predictions', total=len(self.gt_langs)):
        #     self.export_dictionary_to_html(lang, f'data/6_results/dict_{lang}.html')

        self.load_model(path_1)

        threshold = 0.999
        self.test(threshold=threshold, compute_ideal_threshold=False, json_result_path=json_result_path)
        # self.evaluate(plots=['explain false positives'], threshold=threshold,
        #               compute_ideal_threshold=False)

        # self.predict_for_languages(self.gt_langs, threshold)

        # if "th-" not in path_1:
        #     path_2 = path_1[
        #              :-4] + f"_precision{test_precision:.2f}_F1{test_f1:.2f}_th{ideal_threshold:.2f}_{'+'.join(self.dc.target_langs)}.bin"
        #     self.save_model(path_2)
        #     os.remove(os.path.join('data/3_models/', path_1))
        #     return path_2
        # return path_1

        # for lang in self.target_langs:
        #     print(f'Predicting for {lang}...')
        #     self.build_dataset(additional_target_lang=lang)
        #     predicted_words_by_qid_node = self.predict_for_languages([lang], threshold)
        #     self.export_dictionary_to_html(lang, f'data/6_results/dict_{lang}.html', predicted_words_by_qid_node)

    def run_gnn_pipeline(self):
        print('Running GNN pipeline...')
        self.build_dataset()
        self.init_model(self.data)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config[
            'learning_rate'], weight_decay=self.config['weight_decay'])
        assert (self.data.is_undirected())

        self.train_link_predictor()

        # path_1 = f"model_{wandb.run.name}.bin"
        self.save_model()


def setup(mag_path, output_model_file=None, output_data_split_file=None, state_files_path='data/0_state',
          use_wandb=True):
    config = {
            "batch_size": 6000,  # 5000 : DNAConv # 6000 : GCNConv   # 10000 # higher batch size --> higher recall
            "epochs": 1000,

            "bias": True,  # works much better than False
            "min_alignment_count": 1,  # 4,
            "min_edge_weight": 0.2,
            "patience": 5,
            "warm_up": 30,  # 10,

            "optimizer": "adam",
            "learning_rate": 0.05,  # 0.005,
            "weight_decay": 0.0,  # 0.005,
            "plot_path": "data/8_plots/",
    }
    if use_wandb:
        wandb.init(
            project="GCN node prediction (link semantic domains to target words), 20 langs dataset",
            config=config)

    dc = LinkPredictionDictionaryCreator(['bid-eng-web'], state_files_path=state_files_path)
    execution_env = ExecutionEnvironment(dc,
                                         gt_langs={'eng', 'fra', 'ind', 'por', 'swh', 'spa', 'hin', 'mal', 'npi', 'ben',
                                                   'mkn', 'cmn'},
                                         config=config,
                                         # graph_path=f'data/7_graphs/graph-nep_min_alignment_count_{wandb.config["min_alignment_count"]}_min_edge_weight_{wandb.config["min_edge_weight"]}_90_12_langs_no-stopword-removal_{suffix}.cpickle')
                                         graph_path=mag_path,
                                         model_path=output_model_file,
                                         dataset_path=output_data_split_file,
                                         wandb_original_run_name=wandb.run.name if use_wandb else 'no-wandb-run')

    # print the torch seed for reproducibility
    print(f'Torch seed: {torch.initial_seed()}')
    # torch.manual_seed(0)
    return execution_env


def refine_mag(input_mag_path, output_mag_file):
    execution_env = setup(output_mag_file, state_files_path=input_mag_path, use_wandb=False)
    execution_env.build_network()


def train(mag_path, output_model_file, output_data_split_file):
    execution_env = setup(mag_path, output_model_file, output_data_split_file)
    execution_env.run_gnn_pipeline()


def evaluate(mag_path, model_file, data_split_file, results_output_file):
    execution_env = setup(mag_path, model_file, data_split_file)
    execution_env.evaluate_model(model_file, data_split_file, results_output_file)
