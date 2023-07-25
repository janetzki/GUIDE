import argparse
import gc
import itertools
import os
import random
import re
from collections import defaultdict, deque
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool

import _pickle as cPickle
import editdistance as editdistance
import math
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
import yaml
from matplotlib.patches import Patch
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score
from torch.nn import Dropout, Linear
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_networkx
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
        fp = ((1 - y_true) * y_pred)
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


class Precision_Loss(torch.nn.Module):
    # Like F1_Loss, but with a higher weight on precision

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
        fp = ((1 - y_true) * y_pred)
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

        precision_loss = (2 * precision + recall) / 3
        precision_loss = precision_loss.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - precision_loss


class FalsePN_Loss(torch.nn.Module):
    """ Count the number of soft false positives and false negatives. """

    def __init__(self, pos_weight=1.0, precision_reward=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.precision_reward = precision_reward

    def forward(self, y_pred, y_true, ):
        y_pred = F.sigmoid(y_pred)

        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        return fp + fn * self.pos_weight / self.precision_reward


f1_loss_function = F1_Loss().cuda()
precision_loss_function = Precision_Loss().cuda()


def get_neg_edges_for_sd(word_idxs, forbidden_edges, sd_idx):
    assert type(forbidden_edges) is set
    neg_edges = list()
    for word_idx in word_idxs:
        if (sd_idx, word_idx) not in forbidden_edges \
                and (sd_idx, word_idx) not in forbidden_edges:
            neg_edges.append((sd_idx, word_idx))
    return neg_edges


class GNNEncoder(torch.nn.Module):
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py
    def __init__(self, hidden_channels, out_channels, dropout, metadata):
        super().__init__()
        # self.dropout = Dropout(dropout)

        # heterogenous graph
        # self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # self.conv2 = SAGEConv((-1, -1), out_channels)

        self.conv1 = GATv2Conv((-1, -1), out_channels, dropout=dropout, metadata=metadata, add_self_loops=False)
        # self.conv2 = GATv2Conv((-1, -1), out_channels, dropout=dropout, metadata=metadata, add_self_loops=False)
        # self.conv3 = SAGEConv((-1, -1), out_channels)

        # self.conv1 = FastRGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        # self.conv2 = FastRGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        # self.conv3 = FastRGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, x_dict, edge_index_dict):
        # z_dict = self.dropout(x_dict)
        # z_dict = self.conv1(x_dict, edge_index_dict).relu()
        # z_dict = self.conv2(z_dict, edge_index_dict)  # .relu()
        # z_dict = self.conv3(z_dict, edge_index_dict)  # .relu()

        z_dict = self.conv1(x_dict, edge_index_dict).relu()
        # z_dict = self.conv2(z_dict, edge_index_dict)

        # z = self.lin(z)
        return z_dict


class EdgeDecoder(torch.nn.Module):
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.lin1 = Linear(1624 + hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x_dict, z_dict, edge_label_index):
        sd_idxs, word_idxs = edge_label_index
        word_embeddings = z_dict['word'][word_idxs]
        word_embeddings = self.dropout(word_embeddings)
        sd_representation = x_dict['semantic_domain'][sd_idxs]  # torch.zeros((len(sd_idxs), 1624)).to('cuda')
        z = torch.cat([sd_representation, word_embeddings], dim=-1)
        # ml z = torch.cat([z_dict['user'][sd_idx], z_dict['movie'][word_idx]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)
        # src, dst = edge_label_index
        # return (z[src] * z[dst]).sum(dim=-1)  # product of a pair of nodes on each edge


class Model(torch.nn.Module):
    # https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        assert num_layers == 1
        assert dropout == 0.0
        # self.dropout = Dropout(dropout)
        # self.encoder = GNNEncoder(hidden_channels, out_channels, dropout, metadata)
        # self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        # self.decoder = EdgeDecoder(out_channels, dropout)

        self.conv1 = GCNConv(in_channels, out_channels, add_self_loops=False, normalize=False, improved=True)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False, improved=True)
        # self.conv3 = GCNConv(hidden_channels, out_channels, add_self_loops=False, improved=True)

        # self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

        # self.conv1 = TransformerConv(in_channels, hidden_channels, heads=1, dropout=dropout)
        # self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        # self.conv3 = TransformerConv(hidden_channels, out_channels, heads=1, dropout=dropout)

        # self.conv1 = GATConv(in_channels, hidden_channels, heads=1, dropout=dropout) # supports no edge_weight
        # self.conv2 = GATConv(hidden_channels, out_channels, heads=1, dropout=dropout)

        # todo: try out DNAConv https://github.com/pyg-team/pytorch_geometric/blob/ae84a38f14591ba9b8ce64e704e04ea1271c3b78/examples/dna.py#L8
        # self.conv1 = DNAConv(in_channels, heads=1, dropout=dropout, add_self_loops=False)

        # Smart initialize weight matrix
        torch.nn.init.zeros_(self.conv1.lin.weight)  # todo: try without this line (random init)
        offset = in_channels - out_channels
        with torch.no_grad():
            for x in range(out_channels):
                # Set diagonal (identity matrix for the question feature)
                self.conv1.lin.weight[x][x + offset] = 50.0  # 3.0
                # # Set (weighted) node degree feature
                # self.conv1.lin.weight[x][0] = 0.01
                # self.conv1.lin.weight[x][1] = 0.01
        # Initialize bias vector
        with torch.no_grad():
            torch.nn.init.constant_(self.conv1.bias, -5.0)  # -3.0)
        self.visualize_weights('initial')

    def forward(self, x, edge_index, edge_weight):
        # z = self.encode(x, edge_index, edge_weight)
        # return self.decode(z, edge_label_index)

        # z_dict = self.encoder(x_dict, edge_index_dict)
        # return self.decoder(x_dict, z_dict, edge_label_index)

        # z = self.dropout(x)
        z = self.conv1(x, edge_index, edge_weight)  # .relu()
        # z = self.conv2(z, edge_index)

        # debugging:
        # x = torch.zeros(1631).to('cuda')
        # x[100] = 1
        # print(self.lin(x))

        # z = self.lin(x)
        return z

    def visualize_weights(self, title):
        # set size to 2000 x 2000 pixels
        plt.rcParams["figure.figsize"] = (20, 20)

        weight_matrix = self.conv1.lin.weight.detach().cpu()
        bias_matrix = self.conv1.bias.detach().cpu()

        # Convert the matrix tensors to a NumPy array
        weight_matrix_np = weight_matrix.numpy()
        bias_matrix_np = bias_matrix.numpy()

        # Add 1 dimension to the vectors
        bias_matrix_np = np.expand_dims(bias_matrix_np, axis=0)

        # Create heatmaps using Matplotlib
        plt.imshow(weight_matrix_np, cmap='rocket', interpolation='nearest')
        plt.colorbar()
        plt.title('Weight Matrix Heatmap ' + title)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()

        # bias has only one dimension, so we have to make it a 2D array to plot it

        bias_matrix_np = np.repeat(bias_matrix_np, repeats=100, axis=0)
        plt.imshow(bias_matrix_np, cmap='rocket', interpolation='nearest')
        plt.colorbar()
        plt.title('Bias Vector Heatmap ' + title)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()

        # reset size to 400 x 400 pixels
        plt.rcParams["figure.figsize"] = (4, 4)


class EarlyStopper:
    # https://stackoverflow.com/a/73704579/8816968
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class LinkPredictor(object):
    # TARGET_EDGE_TYPE = ('semantic_domain', 'has', 'word')
    # ml TARGET_EDGE_TYPE = ('user', 'rates', 'movie')

    def __init__(self, dc, target_langs, config, graph_path):
        self.dc = dc
        # self.source_langs = set() # {'eng'}
        self.target_langs = target_langs
        self.all_langs = dc.target_langs
        self.gt_langs = self.target_langs  # self.source_langs.union(self.target_langs)
        # assert len(self.source_langs.intersection(self.target_langs)) == 0  # disjoint sets

        self.node_labels_training = list(self.gt_langs)
        self.node_labels_training.append('semantic_domain')
        self.node_labels_all = list(self.all_langs)
        self.node_labels_all.append('semantic_domain')

        self.graph_path = graph_path
        self.G = None

        self.sd_node_idx_by_name = None
        self.word_node_idx_by_name = None
        self.sd_node_names = None
        self.word_node_names = None
        self.word_node_idxs_by_lang = None
        self.word_node_names_by_sd_name = None
        self.color_by_node = None

        self.config = config
        self.model = None
        self.optimizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.palette = None
        self.forbidden_neg_train_edges = None
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_loaders = None
        self.color_masks = []
        self.empty_questions_by_lang = None

    def build_example_network(self):
        self.G = nx.Graph()

        # ----------------
        # ADD NODES
        # ----------------
        # Add target words
        self.G.add_node("mond", lang="deu")
        self.G.add_node("satellit", lang="deu")
        self.G.add_node("stern", lang="deu")
        self.G.add_node("sternenlicht", lang="deu")
        self.G.add_node("wandelstern", lang="deu")
        self.G.add_node("merkur", lang="deu")
        self.G.add_node("klar", lang="deu")
        self.G.add_node("wetter", lang="deu")
        self.G.add_node("windig", lang="deu")
        self.G.add_node("sturm", lang="deu")
        self.G.add_node("wolke", lang="deu")
        self.G.add_node("schatten", lang="deu")
        self.G.add_node("regen", lang="deu")
        self.G.add_node("niederschlag", lang="deu")
        self.G.add_node("feuchtigkeit", lang="deu")
        self.G.add_node("wasserdicht", lang="deu")
        self.G.add_node("botanik", lang="deu")
        self.G.add_node("botaniker", lang="deu")
        self.G.add_node("zahm", lang="deu")
        self.G.add_node("domestiziert", lang="deu")
        self.G.add_node("katze", lang="deu")
        self.G.add_node("hund", lang="deu")

        # Add nodes for semantic domains
        self.G.add_node("Universe", lang="semantic_domain", index="1", size=5)
        self.G.add_node("Sky", lang="semantic_domain", index="1.1", size=4)
        self.G.add_node("Sun", lang="semantic_domain", index="1.1.1", size=3)
        self.G.add_node("Moon", lang="semantic_domain", index="1.1.1.1", size=2)
        self.G.add_node("Star", lang="semantic_domain", index="1.1.1.2", size=2)
        self.G.add_node("Planet", lang="semantic_domain", index="1.1.1.3", size=2)
        self.G.add_node("Weather", lang="semantic_domain", index="1.1.3", size=3)
        self.G.add_node("Wind", lang="semantic_domain", index="1.1.3.1", size=2)
        self.G.add_node("Cloud", lang="semantic_domain", index="1.1.3.2", size=2)
        self.G.add_node("Rain", lang="semantic_domain", index="1.1.3.3", size=2)
        self.G.add_node("Water", lang="semantic_domain", index="1.3", size=3)
        self.G.add_node("Plant", lang="semantic_domain", index="1.5", size=4)
        self.G.add_node("Animal", lang="semantic_domain", index="1.6", size=4)
        self.G.add_node("Types of animals", lang="semantic_domain", index="1.6.1", size=3)
        self.G.add_node("Mammal", lang="semantic_domain", index="1.6.1.1", size=2)
        self.G.add_node("Carnivore", lang="semantic_domain", index="1.6.1.1.2", size=1)

        # Add source words
        self.G.add_node("moon", lang="eng")
        self.G.add_node("satellite", lang="eng")
        self.G.add_node("star", lang="eng")
        self.G.add_node("starlight", lang="eng")
        self.G.add_node("wandering star", lang="eng")
        self.G.add_node("mercury", lang="eng")
        self.G.add_node("clear", lang="eng")
        self.G.add_node("weather", lang="eng")
        self.G.add_node("windy", lang="eng")
        self.G.add_node("storm", lang="eng")
        self.G.add_node("cloud", lang="eng")
        self.G.add_node("shadow", lang="eng")
        self.G.add_node("rain", lang="eng")
        self.G.add_node("precipitation", lang="eng")
        self.G.add_node("moisture", lang="eng")
        self.G.add_node("waterproof", lang="eng")
        self.G.add_node("botany", lang="eng")
        self.G.add_node("botanist", lang="eng")
        self.G.add_node("tame", lang="eng")
        self.G.add_node("domesticated", lang="eng")
        self.G.add_node("cat", lang="eng")
        self.G.add_node("dog", lang="eng")

        # ----------------
        # ADD EDGES
        # ----------------
        # Add edges for semantic domain membership
        self.G.add_edge("moon", "Moon", relation="S")
        self.G.add_edge("satellite", "Planet", relation="S")
        self.G.add_edge("star", "Star", relation="S")
        self.G.add_edge("starlight", "Star", relation="S")
        self.G.add_edge("wandering star", "Planet", relation="S")
        self.G.add_edge("mercury", "Planet", relation="S")
        self.G.add_edge("clear", "Weather", relation="S")
        self.G.add_edge("weather", "Weather", relation="S")
        self.G.add_edge("windy", "Wind", relation="S")
        self.G.add_edge("storm", "Wind", relation="S")
        self.G.add_edge("cloud", "Cloud", relation="S")
        self.G.add_edge("shadow", "Cloud", relation="S")
        self.G.add_edge("rain", "Rain", relation="S")
        self.G.add_edge("precipitation", "Rain", relation="S")
        self.G.add_edge("moisture", "Water", relation="S")
        self.G.add_edge("waterproof", "Water", relation="S")
        self.G.add_edge("botany", "Plant", relation="S")
        self.G.add_edge("botanist", "Plant", relation="S")
        self.G.add_edge("tame", "Types of animals", relation="S")
        self.G.add_edge("domesticated", "Types of animals", relation="S")
        self.G.add_edge("cat", "Carnivore", relation="S")
        self.G.add_edge("dog", "Carnivore", relation="S")

        self.G.add_edge("mond", "Moon", relation="S")
        self.G.add_edge("satellit", "Planet", relation="S")
        self.G.add_edge("stern", "Star", relation="S")
        self.G.add_edge("sternenlicht", "Star", relation="S")
        self.G.add_edge("wandelstern", "Planet", relation="S")
        self.G.add_edge("merkur", "Planet", relation="S")
        self.G.add_edge("wetter", "Weather", relation="S")
        self.G.add_edge("klar", "Weather", relation="S")
        self.G.add_edge("windig", "Wind", relation="S")
        self.G.add_edge("sturm", "Wind", relation="S")
        self.G.add_edge("wolke", "Cloud", relation="S")
        self.G.add_edge("schatten", "Cloud", relation="S")
        self.G.add_edge("regen", "Rain", relation="S")
        self.G.add_edge("niederschlag", "Rain", relation="S")
        self.G.add_edge("feuchtigkeit", "Water", relation="S")
        self.G.add_edge("wasserdicht", "Water", relation="S")
        self.G.add_edge("botanik", "Plant", relation="S")
        self.G.add_edge("botaniker", "Plant", relation="S")
        self.G.add_edge("zahm", "Types of animals", relation="S")
        self.G.add_edge("domestiziert", "Types of animals", relation="S")
        self.G.add_edge("katze", "Carnivore", relation="S")
        self.G.add_edge("hund", "Carnivore", relation="S")

        # Add edges for semantic domain hierarchy
        self.G.add_edge("Universe", "Sky", relation="S")
        self.G.add_edge("Sky", "Sun", relation="S")
        self.G.add_edge("Sun", "Moon", relation="S")
        self.G.add_edge("Sun", "Star", relation="S")
        self.G.add_edge("Sun", "Planet", relation="S")
        self.G.add_edge("Sky", "Weather", relation="S")
        self.G.add_edge("Weather", "Wind", relation="S")
        self.G.add_edge("Weather", "Cloud", relation="S")
        self.G.add_edge("Weather", "Rain", relation="S")
        self.G.add_edge("Universe", "Plant", relation="S")
        self.G.add_edge("Universe", "Animal", relation="S")
        self.G.add_edge("Universe", "Water", relation="S")
        self.G.add_edge("Animal", "Types of animals", relation="S")
        self.G.add_edge("Types of animals", "Mammal", relation="S")
        self.G.add_edge("Mammal", "Carnivore", relation="S")

        # Add edges for word alignments
        self.G.add_edge("mond", "moon", relation="A")
        self.G.add_edge("satellit", "satellite", relation="A")
        self.G.add_edge("stern", "star", relation="A")
        self.G.add_edge("sternenlicht", "starlight", relation="A")
        self.G.add_edge("wandelstern", "wandering star", relation="A")
        self.G.add_edge("merkur", "mercury", relation="A")
        self.G.add_edge("wetter", "weather", relation="A")
        self.G.add_edge("klar", "clear", relation="A")
        self.G.add_edge("windig", "windy", relation="A")
        self.G.add_edge("sturm", "storm", relation="A")
        self.G.add_edge("wolke", "cloud", relation="A")
        self.G.add_edge("schatten", "shadow", relation="A")
        self.G.add_edge("regen", "rain", relation="A")
        self.G.add_edge("niederschlag", "precipitation", relation="A")
        self.G.add_edge("feuchtigkeit", "moisture", relation="A")
        self.G.add_edge("wasserdicht", "waterproof", relation="A")
        self.G.add_edge("botanik", "botany", relation="A")
        self.G.add_edge("botaniker", "botanist", relation="A")
        self.G.add_edge("zahm", "tame", relation="A")
        self.G.add_edge("domestiziert", "domesticated", relation="A")
        self.G.add_edge("katze", "cat", relation="A")
        self.G.add_edge("hund", "dog", relation="A")

        return self.G

    def build_network(self):
        # load graph from file if it exists
        if os.path.exists(self.graph_path):
            print(f'Loading graph from {self.graph_path}')
            with open(self.graph_path, 'rb') as f:
                gc.disable()  # disable garbage collection to speed up loading
                self.G, self.empty_questions_by_lang = cPickle.load(f)
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
            self.G.add_node(long_qid_name_by_qid[qid], lang='semantic_domain')
            # if len(self.G) == 50:
            #     break

        # add nodes for words and edges to semantic domains and alignments
        nodes_by_lang = defaultdict(set)
        edges = set()
        for lang in tqdm(self.all_langs,
                         desc='Building network',
                         total=len(self.all_langs)):
            for word in self.dc.words_by_text_by_lang[lang].values():
                # is_connected_to_semantic_domain = False

                # add word-qid edges
                for qid in word.qids:
                    if qid.startswith('9'):
                        continue
                    long_qid_name = long_qid_name_by_qid[qid]
                    # if long_qid_name not in self.G.nodes:  # for 50 qids etc.
                    #     continue
                    assert long_qid_name in self.G.nodes
                    edges.add((str(word), long_qid_name, 1))
                    # is_connected_to_semantic_domain = True

                # if not is_connected_to_semantic_domain:
                #     continue

                nodes_by_lang[lang].add(str(word))

                # add word-word edges
                for translation, alignment_count in word.get_aligned_words_and_counts(
                        self.dc.words_by_text_by_lang):
                    translation_lang = translation.iso_language
                    if translation_lang == lang:
                        # avoid self-loops (for English words)
                        continue
                    nodes_by_lang[translation_lang].add(str(translation))

                    # # skip words that only belong to qids that are not in the graph
                    # if not any(long_qid_name_by_qid[qid] in self.G.nodes for qid in translation.qids):
                    #     continue

                    # skip rare edges for higher precision and speed
                    if alignment_count < 4:
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

        self.normalize_edge_weights()
        self.filter_edges_by_weight()
        self.find_empty_questions_by_lang(long_qid_name_by_qid)

        # remove target language nodes with no edge to a gt lang node because we cannot predict their semantic domain
        for lang in self.all_langs:
            for node in nodes_by_lang[lang]:
                if len([n for n in self.G.neighbors(node) if self.G.nodes[n]['lang'] in self.gt_langs]) == 0:
                    self.G.remove_node(node)

        print(f'Saving graph to {self.graph_path}')
        with open(self.graph_path, 'wb') as f:
            cPickle.dump((self.G, self.empty_questions_by_lang), f)
        print('Done saving graph')

    def normalize_edge_weights(self):
        for edge in tqdm(self.G.edges(data='weight'), desc='Normalizing edge weights', total=len(self.G.edges)):
            u, v = edge[0], edge[1]
            lang1 = self.G.nodes[u]['lang']
            lang2 = self.G.nodes[v]['lang']
            if 'semantic_domain' in (lang1, lang2):
                continue
            word1 = self.dc.words_by_text_by_lang[lang1][u.split(': ')[1]]
            word2 = self.dc.words_by_text_by_lang[lang2][v.split(': ')[1]]
            self.G[u][v]['weight'] = self.dc._compute_link_score(word1, word2)

    def filter_edges_by_weight(self):
        # filter all edges with weight < 0.2
        removed_edges = []
        for edge in tqdm(self.G.edges(data='weight'), desc='Filtering edges by weight', total=len(self.G.edges)):
            if edge[2] < 0.2:
                removed_edges.append(edge)
        for edge in removed_edges:
            self.G.remove_edge(edge[0], edge[1])

    def find_empty_questions_by_lang(self, long_qid_name_by_qid):
        # for each lang, find all qids that have no edge to a word in that lang
        empty_questions_by_lang = defaultdict(set)
        for lang in tqdm(self.gt_langs, desc='Finding empty questions by lang', total=len(self.gt_langs)):
            for long_qid_name in long_qid_name_by_qid.values():
                neighbors = [n for n in self.G.neighbors(long_qid_name) if self.G.nodes[n]['lang'] == lang]
                if len(neighbors) == 0:
                    empty_questions_by_lang[lang].add(long_qid_name)
        self.empty_questions_by_lang = dict(empty_questions_by_lang)

    def convert_empty_questions_by_lang_to_tensors(self):
        for lang in self.gt_langs:
            temp = torch.zeros(len(self.sd_node_idx_by_name), dtype=torch.bool)
            for long_qid_name in self.empty_questions_by_lang[lang]:
                qid_idx = self.sd_node_idx_by_name[long_qid_name]
                temp[qid_idx] = True
            self.empty_questions_by_lang[lang] = temp

    def plot_subgraph(self, graph, node):
        # filtered_word_nodes = [word for word in graph.nodes() if word.iso_language in target_langs]
        #
        # filtered_weighted_edges = []
        # for edge in graph.edges(data='weight'):
        #     lang_1 = edge[0].iso_language
        #     lang_2 = edge[1].iso_language
        #     wtxt_1 = edge[0].text
        #     wtxt_2 = edge[1].text
        #     count = edge[2]
        #     if lang_1 not in target_langs or lang_2 not in target_langs \
        #             or (lang_1 == lang_2 and wtxt_1 == wtxt_2) \
        #             or (count < min_count and self._compute_link_score(edge[0], edge[1]) < score_threshold):
        #         continue
        #     filtered_weighted_edges.append(edge)
        #
        # filtered_word_graph = nx.Graph()
        # filtered_word_graph.add_nodes_from(filtered_word_nodes)
        # filtered_word_graph.add_weighted_edges_from(filtered_weighted_edges)

        # define filtered subgraph of a node's 1st, 2nd, and 3rd order neighbors
        selected_nodes = {node}
        neighbors_1st_order = set()
        neighbors_2nd_order = set()
        neighbors_3rd_order = set()
        for neighbor_1st_order in graph.neighbors(node):
            neighbors_1st_order.add(neighbor_1st_order)
            for neighbor_2nd_order in graph.neighbors(neighbor_1st_order):
                neighbors_2nd_order.add(neighbor_2nd_order)
                for neighbor_3rd_order in graph.neighbors(neighbor_2nd_order):
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
                selected_nodes.update([n for n in neighbors_3rd_order if graph.nodes[n]['lang'] == 'semantic_domain'])
        else:
            # only add semantic domain nodes
            selected_nodes.update([n for n in neighbors_2nd_order if graph.nodes[n]['lang'] == 'semantic_domain'])
        displayed_subgraph = graph.subgraph(selected_nodes)
        assert (len(displayed_subgraph.nodes) <= len(
            displayed_subgraph.edges) + 1)  # necessary condition if graph is connected

        # # if nodes > max_nodes, remove cmn nodes
        # if len(displayed_subgraph.nodes) > max_nodes:
        #     selected_nodes = [node for node in displayed_subgraph.nodes if
        #                  graph.nodes[node]['lang'] != 'cmn']
        #     displayed_subgraph = displayed_subgraph.subgraph(selected_nodes)

        # set figure size heuristically
        width = max(6, int(len(selected_nodes) / 2.2))
        plt.figure(figsize=(width, width))

        # use a different node color for each language
        node_colors = [self.palette[data['lang']] for node_name, data in displayed_subgraph.nodes(data=True)]

        # show all the colors in a legend
        plt.legend(handles=[Patch(color=self.palette[lang], label=lang) for lang in self.node_labels_all])

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(displayed_subgraph)

        # draw nodes
        nx.draw_networkx_nodes(displayed_subgraph, pos=pos, node_color=node_colors)

        # draw only word texts as node labels
        nx.draw_networkx_labels(displayed_subgraph, pos=pos,
                                labels={node_name: node_name.split(': ')[1] for node_name in
                                        displayed_subgraph.nodes()})

        # draw edges (thicker edges for more frequent alignments)
        for edge in displayed_subgraph.edges(data='weight'):
            weight = edge[2] if edge[2] is not None else 1
            nx.draw_networkx_edges(displayed_subgraph, pos=pos, edgelist=[edge],
                                   # caution: might fail in debug mode with Python 3.10 instead of Python 3.9 or 3.11
                                   width=[math.log(weight * 10) + 1], alpha=0.5)

        # draw edge labels with 2 decimal places
        edge_weights = nx.get_edge_attributes(displayed_subgraph, 'weight')
        edge_labels = dict([((u, v), f"{d['weight']:.2f}") for u, v, d in displayed_subgraph.edges(data=True)])
        # skip edges to semantic domains
        edge_labels = {k: v for k, v in edge_labels.items() if
                       displayed_subgraph.nodes[k[0]]['lang'] != 'semantic_domain' and
                       displayed_subgraph.nodes[k[1]]['lang'] != 'semantic_domain'}
        if len(edge_weights):
            nx.draw_networkx_edge_labels(displayed_subgraph, pos, edge_labels=edge_labels)

        plt.title(f'Nodes close to "{node}"')
        plt.show()

    def plot_graph_with_labels(self, graph, title, node_sizes, node_colors, all_edges, target_edges,
                               n_sample=1000):
        # sample nodes, node_sizes, and node_colors
        n_sample = min(n_sample, len(graph.nodes))
        sampled_nodes = random.sample(self.G.nodes, n_sample)
        sampled_graph = self.G.subgraph(sampled_nodes)
        sampled_graph = nx.Graph(sampled_graph)  # unfreeze graph to allow for node adding
        sampled_node_sizes = [node_sizes[self.node_idx_by_type_by_name[node]] for node in sampled_graph.nodes]
        sampled_node_colors = [node_colors[self.node_idx_by_type_by_name[node]] for node in sampled_graph.nodes]

        # set figure size heuristically
        width = max(6, int(len(sampled_graph.nodes()) / 2.2))
        plt.figure(figsize=(width, width))
        plt.legend(handles=[Patch(color=self.palette[lang], label=lang) for lang in self.node_labels_all])

        graph_edges = graph.edges
        if graph is not self.G:
            # convert edges between ints to edges between node names
            all_edges = [(self.node_name_by_index[edge[0]], self.node_name_by_index[edge[1]]) for edge in all_edges]
            target_edges = [(self.node_name_by_index[edge[0]], self.node_name_by_index[edge[1]]) for edge in
                            target_edges]
            graph_edges = [(self.node_name_by_index[edge[0]], self.node_name_by_index[edge[1]]) for edge in graph_edges]

        highlighted_positive_edges = [edge for edge in target_edges if
                                      edge in graph_edges or (edge[1], edge[0]) in graph_edges]
        highlighted_negative_edges = []
        if graph is not self.G:
            highlighted_negative_edges = [edge for edge in graph_edges if
                                          edge not in all_edges and (edge[1], edge[0]) not in all_edges]
            for edge in all_edges:
                if edge[0] in sampled_graph and edge[1] in sampled_graph:
                    sampled_graph.add_edge(edge[0],
                                           edge[1])  # add all edges to the graph, so that they are also plotted
        edge_labels = nx.get_edge_attributes(graph, "relation")
        pos = graphviz_layout(sampled_graph)  # , k=0.3, iterations=20)

        # Draw nodes with size
        nx.draw_networkx_nodes(sampled_graph, pos, node_color=sampled_node_colors, node_size=sampled_node_sizes)
        nx.draw_networkx_labels(sampled_graph, pos=pos,
                                labels={node_name: node_name.split(': ')[1] for node_name in sampled_graph.nodes()})
        nx.draw_networkx_edges(sampled_graph, pos, edgelist=sampled_graph.edges,
                               alpha=0.5 if 'decoder' in title else 1.0)
        nx.draw_networkx_edge_labels(sampled_graph, pos, edge_labels=edge_labels)

        # also draw edges that are not in the graph
        nx.draw_networkx_edges(sampled_graph, pos, edgelist=highlighted_positive_edges, edge_color="green", width=3)
        nx.draw_networkx_edges(sampled_graph, pos, edgelist=highlighted_negative_edges, edge_color="red", width=3)

        plt.title(title)
        plt.show()

    def convert_to_networkx(self, graph, n_sample=100):
        g = to_networkx(graph, node_attrs=["x"], to_undirected=True)
        #    y = graph.y.numpy()

        if len(g.nodes) > n_sample:
            sampled_nodes = random.sample(g.nodes, n_sample)
            g = g.subgraph(sampled_nodes)
        #        y = y[sampled_nodes]

        return g  # , y

    def add_backward_edges(self, data):
        # todo: use T.ToUndirected() instead or don't use this at all
        data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)

        # deduplicate edges
        data.edge_index = torch.unique(data.edge_index, dim=1)

        # build edge weights
        data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float32)
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[:, i]
            u = u.item()
            v = v.item()
            edge_weight = self.G.edges[(self.node_name_by_index[u], self.node_name_by_index[v])].get("weight")
            data.edge_weight[i] = edge_weight

    def convert_edge_label_index_to_networkx(self, graph):
        graph_copy = graph.clone()
        graph_copy.edge_index = graph_copy.edge_label_index
        self.add_backward_edges(graph_copy)
        g = to_networkx(graph_copy, node_attrs=["x"], to_undirected=True)
        return g

    def l1_regularization(self, model):
        # https://stackoverflow.com/a/66543549/8816968
        l1_lambda = 0.001
        l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
        return l1_lambda * l1_norm

    def train(self):
        ys = []
        y_hats = []
        # masks = []
        node_idxs = []
        self.model.train()
        for train_loader in self.train_loaders:
            for batch in train_loader:
                #     assert batch.node_stores[0].num_nodes == 1624
                # for mask in tqdm(self.color_masks):
                self.optimizer.zero_grad()
                # batch = self.data
                batch.to(self.device)
                # mask = batch.train_mask # & mask

                # neg_sampling_ratio = 100.0
                # num_pos_edges = len(self.train_data[self.TARGET_EDGE_TYPE].edge_label)
                # num_neg_edges = neg_sampling_ratio * num_pos_edges
                # neg_edges = self.sample_neg_edges(num_neg_edges, self.forbidden_neg_train_edges)
                # self.add_negative_edges(self.train_data, neg_edges)

                x = batch.x.clone()
                edge_index = batch.edge_index
                # edge_label_index = batch[self.TARGET_EDGE_TYPE].edge_label_index
                edge_weight = batch.edge_weight
                y = batch.y[:batch.batch_size]

                x[:batch.batch_size] = 0  # mask out the nodes to be predicted

                # y_hat = model(x, edge_index) # .index_select(0, torch.tensor(train_indices).to(self.device))

                # z = self.model.encode(x, edge_index, edge_weight, edge_type)
                # y_hat = self.model.decode(z, edge_label_index).view(-1)
                y_hat = self.model(x, edge_index, edge_weight)[:batch.batch_size]

                train_loss = self.compute_loss_with_empty_questions(y_hat, y, batch.n_id[:batch.batch_size])

                # link_probs = out.sigmoid()
                # total_auc += roc_auc_score(y.detach().cpu().numpy(), link_probs.detach().cpu().numpy())
                ys.append(y.detach().cpu())
                y_hats.append(y_hat.detach().cpu())
                node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())
                # masks.append(mask.detach().cpu())

                # train_mirco_f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
                # train_accuracy = accuracy_score(y.cpu(), preds.cpu())

                train_loss.backward()
                self.optimizer.step()

                # Log the weights, biases and gradients for each layer of the model
                # self.writer = SummaryWriter()
                # for name, param in self.model.named_parameters():
                #     self.writer.add_histogram(name, param, epoch)
                #     self.writer.add_histogram(f'{name}.grad', param.grad, epoch)

                # # Remove negative edges from train_data
                # self.train_data[self.TARGET_EDGE_TYPE].edge_label = self.train_data[self.TARGET_EDGE_TYPE].edge_label[:num_pos_edges]
                # self.train_data[self.TARGET_EDGE_TYPE].edge_label_index = self.train_data[self.TARGET_EDGE_TYPE].edge_label_index[:, :num_pos_edges]

        y = torch.cat(ys, dim=0)
        y_hat = torch.cat(y_hats, dim=0)
        node_idxs = torch.cat(node_idxs, dim=0)
        assert (len(y) == len(y_hat))
        assert (len(y) == self.data.train_mask.sum())
        train_loss = self.compute_loss_with_empty_questions(y_hat.to(self.device), y.to(self.device), node_idxs)

        # y = y.cpu().numpy()
        # y_hat = y_hat.cpu().numpy()
        # mask = torch.cat(masks, dim=0).detach().cpu().numpy()
        # y_flat = y.flatten()
        # y_hat_flat = y_hat.flatten()
        train_auc = -1.0  # roc_auc_score(y_flat, y_hat_flat)

        return train_loss, train_auc

    def train_link_predictor(self):
        # val_loss, val_auc = eval_link_predictor(model, val_data)
        # print(f"Epoch: {0:03d}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        early_stopper = EarlyStopper(patience=5)
        for epoch in range(1, self.config['train']['epochs'] + 1):
            val_loss, val_auc, _, _, _, precision, recall, _, f1, f1_loss = self.evaluate(
                self.data.val_mask, threshold=0.99, compute_ideal_threshold=False)

            train_loss, train_auc = self.train()
            wandb.log({"train loss": train_loss, "train AUC": train_auc, "val loss": val_loss, "val AUC": val_auc,
                       "val F1 loss": f1_loss})
            wandb.watch(self.model)

            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val F1 Loss: {f1_loss:.6f}")

            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

    def compute_loss_with_empty_questions(self, y_hat, y, node_idxs):
        # do not count candidates at unknown positions
        node_langs = [self.word_node_names[i.item()].split(': ')[0] for i in node_idxs]
        empty_questions = [self.empty_questions_by_lang[lang] for lang in node_langs]
        empty_questions = torch.stack(empty_questions).to(self.device)
        return self.criterion(y_hat, y, empty_questions)

    def compute_metric_with_empty_questions(self, candidates, node_idxs):
        # do not count candidates at unknown positions
        node_langs = [self.word_node_names[i.item()].split(': ')[0] for i in node_idxs]
        empty_questions = [self.empty_questions_by_lang[lang] for lang in node_langs]
        valid_questions = torch.logical_not(torch.stack(empty_questions)).to(self.device)
        candidates = candidates.to(self.device)
        return torch.sum(torch.logical_and(candidates, valid_questions)).item()

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
                                         save_to_csv_path=None, word_idx_map=None, max_print_total=100):
        print("Printing human readable predictions...")
        predictions = []
        # for each element in out, lookup all edges in eval_data[self.TARGET_EDGE_TYPE].edge_label_index
        count_printed_words = 0
        count_printed_total = 0
        assert (mask is None or len(mask) == len(link_probs))
        total_words = mask.sum().item() if mask is not None else len(link_probs)
        for word_idx, sd_preds in tqdm(enumerate(link_probs), total=total_words):
            if mask is not None and not mask[word_idx]:
                continue
            printed = False
            for sd_idx, prob in enumerate(sd_preds):
                label = y[word_idx][sd_idx] if y is not None else None
                pred = 1 if prob >= threshold else 0
                is_correct = label is None or pred == label
                if pred == 0:  # and is_correct:
                    continue

                word = self.word_node_names[word_idx_map[word_idx].item()] if word_idx_map is not None else \
                    self.word_node_names[word_idx]
                sd = self.sd_node_names[sd_idx]
                pred_raw = y_hat[word_idx][sd_idx]

                prefix = '! ' if not is_correct and label is not None else '  '  # highlight false positives and false negatives
                similar_words = []
                if pred == 1 and not is_correct:
                    similar_words = self.find_similar_words_in_sd(word, sd)
                print(
                    f'{prefix} prediction: {pred} ({pred_raw:.0f} --> {prob:.2f}/{threshold:.2f}), actual: {label}, {word} ({word_idx}) <---> {sd} ({sd_idx})',
                    similar_words)
                printed = True
                count_printed_total += 1
                if pred == 1:
                    predictions.append([sd, word, prob.item()])
            count_printed_words += printed
            if count_printed_words >= max_print_words or count_printed_total >= max_print_total:
                print(f'... and {len(link_probs) - max_print_words} more words')
                break
        print(f'Printed {count_printed_words} words and {count_printed_total} total predictions')

        if save_to_csv_path is not None:
            print(f'Saving predictions to {save_to_csv_path}...')
            with open(save_to_csv_path, 'w') as f:
                f.write('qid,semantic domain,word,question,score\n')
                for prediction in predictions:
                    sd_qid_question = prediction[0].split(': ')[1]
                    semantic_domain = re.search(r'^([^0-9]+)', sd_qid_question).group(1).strip()
                    qid = re.search(r'([0-9]+.*?)(?= \()', sd_qid_question).group(1)
                    question = re.search(r'\((.*?)\)', sd_qid_question).group(1)
                    f.write(qid + ',')
                    f.write('"' + semantic_domain + '",')
                    f.write(str(prediction[1]) + ',')
                    f.write('"' + question + '",')
                    f.write(str(prediction[2]) + '\n')
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
    def evaluate(self, mask, plots=None, print_human_readable_predictions=False,
                 compute_additional_metrics=False, threshold=None, compute_ideal_threshold=True):
        assert ((threshold is not None and 0 <= threshold <= 1) or compute_ideal_threshold)
        if plots is None:
            plots = []
        self.model.eval()
        y_hats = []
        ys = []
        node_idxs = []
        # edge_label_indexes = []

        # eval_loader = LinkNeighborLoader(
        #     data=eval_data.to(self.device),
        #     num_neighbors=[-1, -1],
        #     edge_label_index=(
        #         self.TARGET_EDGE_TYPE, eval_data[self.TARGET_EDGE_TYPE].edge_label_index.to(self.device)),
        #     edge_label=eval_data[self.TARGET_EDGE_TYPE].edge_label.to(self.device),
        #     batch_size=self.config['train']['batch_size'],  # number of edges per batch
        #     shuffle=True,
        # )
        eval_loaders = [NeighborLoader(
            self.data,
            num_neighbors=[-1],
            batch_size=self.config['train']['batch_size'],
            input_nodes=mask.to(self.device) & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        ) for color_mask in self.color_masks if (mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        for eval_loader in eval_loaders:
            for batch in eval_loader:
                #         assert batch.node_stores[0].num_nodes == 1624
                #     batch = self.data
                x = batch.x.clone()
                edge_index = batch.edge_index
                # sub_mask = batch.mask
                # edge_label_index = batch[self.TARGET_EDGE_TYPE].edge_label_index
                edge_weight = batch.edge_weight
                # edge_type = eval_data.edge_type
                # assert self.convert_edge_index_to_set_of_tuples(edge_label_index).issubset(
                #     self.convert_edge_index_to_set_of_tuples(eval_loader.data[self.TARGET_EDGE_TYPE].edge_label_index))

                x[:batch.batch_size] = 0  # mask out the nodes to be predicted

                # z = self.model.encode(x, edge_index, edge_weight, edge_type)
                y_hat = self.model(x, edge_index, edge_weight)[
                        :batch.batch_size]  # self.model.decode(z, edge_label_index).view(-1)
                # y_hats.append(self.model(x_dict, edge_index_dict, edge_label_index))
                y_hats.append(y_hat.detach().cpu())
                # ys.append(batch[self.TARGET_EDGE_TYPE].edge_label.to(self.device))
                y = batch.y[:batch.batch_size]
                ys.append(y.detach().cpu())
                # edge_label_indexes.append(edge_label_index)
                node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())

        # mask_lord = torch.zeros(len(self.data.x), dtype=torch.bool)
        # mask_lord[self.word_node_idx_by_name['eng: lord']] = True
        # nl = NeighborLoader(
        #     self.data,
        #     num_neighbors=[-1, -1],
        #     batch_size=1,
        #     input_nodes=mask_lord.to(self.device),
        #     shuffle=True,
        #     subgraph_type='bidirectional',
        # )
        # for batch in nl:
        #     batch.to(self.device)
        #     y_hat_2 = self.model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
        #     pass

        y_hat = torch.cat(y_hats, dim=0)
        y = torch.cat(ys, dim=0)
        node_idxs = torch.cat(node_idxs, dim=0)
        assert y.sum() > 0  # at least one positive example
        assert (len(y) == len(y_hat))
        assert (len(y) == len(node_idxs))
        assert (len(y) == mask.sum())
        # edge_label_index = torch.cat(edge_label_indexes, dim=1)
        # assert self.convert_edge_index_to_set_of_tuples(edge_label_index) == self.convert_edge_index_to_set_of_tuples(eval_loader.data[self.TARGET_EDGE_TYPE].edge_label_index)
        eval_loss = self.compute_loss_with_empty_questions(y_hat.to(self.device), y.to(self.device), node_idxs)
        f1_loss = -1.0  # f1_loss_function(y_hat, y)
        link_probs = y_hat.sigmoid()
        # y = eval_data[self.TARGET_EDGE_TYPE].edge_label.cpu().numpy()
        assert (abs(self.model(self.data.x, self.data.edge_index, self.data.edge_weight)[
                        # assert that the batches lead to the same result, ignoring numerical errors
                        mask].sigmoid().sum().item() - link_probs.sum().item()) < 10.0)
        y_tensor = y.cpu()
        link_probs_tensor = link_probs.cpu()
        y = y.cpu().numpy()
        link_probs = link_probs.cpu().numpy()
        # mask = mask.cpu()
        y_flat = y.flatten()
        link_probs_flat = link_probs.flatten()
        auc = -1.0  # roc_auc_score(y_flat, link_probs_flat) if y_flat.sum() > 0 else 0.0
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
            print(f'AUC: {auc:.2f}')
            num_predictions = len(y_flat)
            print(f'Samples: {mask.sum()} words, {num_predictions} predictions')
            print(f'Ideal threshold: {ideal_threshold:.2f}')
            print(f'Used threshold: {threshold:.2f}')
            false_positives = self.compute_metric_with_empty_questions(
                torch.logical_and(link_probs_tensor >= threshold, y_tensor == 0), node_idxs)
            print(f'False positives: {false_positives} ({false_positives / num_predictions * 100:.2f} %)')
            false_negatives = self.compute_metric_with_empty_questions(
                torch.logical_and(link_probs_tensor < threshold, y_tensor == 1), node_idxs)
            print(f'False negatives: {false_negatives} ({false_negatives / num_predictions * 100:.2f} %)')
            true_positives = self.compute_metric_with_empty_questions(
                torch.logical_and(link_probs_tensor >= threshold, y_tensor == 1), node_idxs)
            print(f'True positives: {true_positives} ({true_positives / num_predictions * 100:.2f} %)')
            true_negatives = self.compute_metric_with_empty_questions(
                torch.logical_and(link_probs_tensor < threshold, y_tensor == 0), node_idxs)
            print(f'True negatives: {true_negatives} ({true_negatives / num_predictions * 100:.2f} %)')
            precision = true_positives / (true_positives + false_positives)
            print(f'Precision: {precision:.2f}')
            recall = true_positives / (true_positives + false_negatives)
            print(f'Recall: {recall:.2f}')
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f'F1: {f1:.2f}')
            # bacc = balanced_accuracy_score(y, y_hat >= ideal_threshold)
            # print(f'Accuracy: {acc:.2f}')

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
            self.model.visualize_weights('after training')

        if print_human_readable_predictions:
            self.print_human_readable_predictions(y, y_hat, link_probs, None, threshold, word_idx_map=node_idxs)

        return eval_loss, auc, ideal_threshold, false_positives, false_negatives, precision, recall, acc, f1, f1_loss

    def explain(self):
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer_link_pred.py
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='edge',
            return_type='raw',
            # task_level='node',
            # return_type='log_probs',
        )

        # Explain model output for a single edge:
        edge_label_index = self.val_data[self.TARGET_EDGE_TYPE].edge_label_index[:, 0]

        explainer = Explainer(  # todo: fix the explainer, explain an edge or at least a node
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
        explanation = explainer(
            x=self.train_data.x.to(self.device),
            edge_index=self.train_data.edge_index.to(self.device),
            edge_weight=self.train_data.edge_weight.to(self.device),
            edge_label_index=edge_label_index.to(self.device),
            index=0,
        )
        print(f'Generated model explanations in {explanation.available_explanations}')

        # Visualize the explanations:
        path = 'plots/explainer/feature_importance_1.png'
        explanation.visualize_feature_importance(path, top_k=10)
        print(f"Feature importance plot has been saved to '{path}'")

        path = 'plots/explainer/subgraph_1.pdf'
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")

        # Explain a selected target (phenomenon) for a single edge:
        edge_label_index = self.val_data[self.TARGET_EDGE_TYPE].edge_label_index[:, 0]
        target = self.val_data[self.TARGET_EDGE_TYPE].edge_label[0].unsqueeze(dim=0).long()

        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
        explanation = explainer(
            x=self.train_data.x.to(self.device),
            edge_index=self.train_data.edge_index.to(self.device),
            edge_weight=self.train_data.edge_weight.to(self.device),
            target=target.to(self.device),
            edge_label_index=edge_label_index.to(self.device),
            index=0,
        )
        available_explanations = explanation.available_explanations
        print(f'Generated phenomenon explanations in {available_explanations}')

        # Visualize the explanations:
        path = 'plots/explainer/feature_importance_2.png'
        explanation.visualize_feature_importance(path, top_k=10)
        print(f"Feature importance plot has been saved to '{path}'")

        path = 'plots/explainer/subgraph_2.pdf'
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")

    @staticmethod
    def convert_edge_index_to_set_of_tuples(edge_index):
        return {(u, v) for u, v in edge_index.transpose(0, 1).tolist()}

    def get_all_neg_edges(self):
        forbidden_edges = self.convert_edge_index_to_set_of_tuples(self.data[self.TARGET_EDGE_TYPE].edge_index)
        func = partial(get_neg_edges_for_sd, self.word_node_idx_by_name.values(), forbidden_edges)
        with Pool(processes=cpu_count()) as pool:
            neg_edges_by_sd = tqdm(pool.imap(func, self.sd_node_idx_by_name.values()),
                                   desc=f'Generating negative edges ({cpu_count()} processes)',
                                   total=len(self.sd_node_idx_by_name))
            return list(itertools.chain(*neg_edges_by_sd))  # flatmap

    def sample_neg_edges(self, num_neg_edges, forbidden_edges=None):
        if forbidden_edges is None:
            forbidden_edges = set()
        forbidden_edges |= self.convert_edge_index_to_set_of_tuples(self.data[self.TARGET_EDGE_TYPE].edge_index)

        # generate edges that are not in the graph and not in val_test_edges
        neg_edges = set()

        # # filter nodes with at least one semantic domain node as neighbor
        # sd_neighbor_nodes_by_lang = defaultdict(set)
        # for lang in self.target_langs:
        #     for node in self.word_node_idxs_by_lang[lang]:
        #         node_name = self.word_node_names[node]
        #         neighbors = [n for n in self.G.neighbors(node_name) if self.G.nodes[n]["lang"] == "semantic_domain"]
        #         if len(neighbors) > 0:
        #             sd_neighbor_nodes_by_lang[lang].add(node)
        # # convert sets to lists
        # sd_neighbor_nodes_by_lang = {lang: list(nodes) for lang, nodes in sd_neighbor_nodes_by_lang.items()}
        #
        # # Generate negative edges for words with connected SD nodes.
        # # Reason: Training with word nodes without connected SDs is too easy.
        # while len(neg_edges) < 0.8 * num_neg_edges:
        #     lang = random.choice(list(self.target_langs))
        #     node1 = random.choice(list(self.sd_node_idx_by_name.values()))
        #     node2 = random.choice(sd_neighbor_nodes_by_lang[lang])
        #     if (node1, node2) not in forbidden_edges and (node2, node1) not in forbidden_edges:
        #         neg_edges.add((node1, node2))

        while len(neg_edges) < num_neg_edges:
            lang = random.choice(list(self.target_langs))
            node1 = random.choice(list(self.sd_node_idx_by_name.values()))
            node2 = random.choice(self.word_node_idxs_by_lang[lang])
            if (node1, node2) not in forbidden_edges and (node2, node1) not in forbidden_edges:
                neg_edges.add((node1, node2))

        return list(neg_edges)

    def add_negative_edges(self, data, neg_edges):
        print(f'Adding {len(neg_edges)} negative edges...')
        data[self.TARGET_EDGE_TYPE].edge_label_index = torch.cat(
            [data[self.TARGET_EDGE_TYPE].edge_label_index, torch.tensor(neg_edges).transpose(0, 1).to(self.device)],
            dim=1)

        # # assert that there is no duplicate in the edge_label_index (no edge is positive and negative or added twice)
        # assert data[self.TARGET_EDGE_TYPE].edge_label_index.size(1) == \
        #        len(self.convert_edge_index_to_set_of_tuples(data[self.TARGET_EDGE_TYPE].edge_label_index))

        data[self.TARGET_EDGE_TYPE].edge_label = torch.cat(
            [data[self.TARGET_EDGE_TYPE].edge_label, torch.zeros(len(neg_edges), dtype=torch.long).to(self.device)])

    def add_negative_edges_to_data_split(self, add_negative_edges_to_train, neg_sampling_ratio=1.0):
        train_edge_label = self.train_data[self.TARGET_EDGE_TYPE].edge_label
        val_edge_label = self.val_data[self.TARGET_EDGE_TYPE].edge_label
        test_edge_label = self.test_data[self.TARGET_EDGE_TYPE].edge_label

        num_pos_edges = len(train_edge_label) + len(val_edge_label) + len(test_edge_label)
        if neg_sampling_ratio == 'all':
            neg_edges = self.get_all_neg_edges()
        else:
            # sample the negative edges
            num_neg_edges = num_pos_edges * neg_sampling_ratio
            neg_edges = self.sample_neg_edges(num_neg_edges)

        # shuffle the negative edges
        random.shuffle(neg_edges)

        # split the negative edges into train, val, and test
        train_num_neg_edges = int(len(train_edge_label) / num_pos_edges * len(neg_edges))
        val_num_neg_edges = int(len(val_edge_label) / num_pos_edges * len(neg_edges))
        train_neg_edges = neg_edges[:train_num_neg_edges]
        val_neg_edges = neg_edges[train_num_neg_edges:train_num_neg_edges + val_num_neg_edges]
        test_neg_edges = neg_edges[train_num_neg_edges + val_num_neg_edges:]

        # add the negative edges to the train, val, and test data
        if add_negative_edges_to_train:
            self.add_negative_edges(self.train_data, train_neg_edges)
        self.add_negative_edges(self.val_data, val_neg_edges)
        self.add_negative_edges(self.test_data, test_neg_edges)
        print(f'Added {len(neg_edges)} negative edges in total.')

    def save_model(self, path):
        print(f'Saving model to {path}')
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        print(f'Loading model from {path}')
        self.init_model(self.data)
        self.model.load_state_dict(torch.load(path))
        self.model.visualize_weights('loaded')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, test_word, threshold):
        # # create dataset
        # semantic_domain_nodes = [node for node in complete_graph.nodes if
        #                          complete_graph.nodes[node]["lang"] == "semantic_domain"]
        #
        # # Feature 1: Language
        # lang_feature = torch.zeros(len(complete_graph), len(self.node_labels_training))
        # for node in complete_graph.nodes:
        #     lang = complete_graph.nodes[node]["lang"]
        #     if lang in langs:
        #         lang = 'fra'
        #     lang_feature[self.node_idx_by_type_by_name[node]][
        #     #     self.node_labels_all.index('eng')] = 1  # complete_graph.nodes[node]["lang"]
        #         self.node_labels_all.index(lang)] = 1
        #
        # # Feature 2: Semantic Domains
        # sd_feature = torch.zeros(len(complete_graph), len(semantic_domain_nodes))
        # for node in complete_graph.nodes:
        #     if complete_graph.nodes[node]["lang"] == "semantic_domain":  # this feature is helpful (tested)
        #         sd_feature[self.node_idx_by_type_by_name[node]][semantic_domain_nodes.index(node)] = 1
        #
        # x = torch.cat([lang_feature, sd_feature], dim=1)
        #
        # # edges to all sds
        # target_edges = [(self.sd_node_idx_by_name[sd], self.word_node_idx_by_name[test_word]) for sd in self.G.nodes if
        #                 self.G.nodes[sd]['lang'] == 'semantic_domain']
        # target_edge_index = torch.tensor(list(target_edges)).transpose(0, 1)
        #
        # num_nodes_by_lang = {
        #     lang: len([node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == lang]) for lang
        #     in self.node_labels_all}
        #
        # pred_data[self.TARGET_EDGE_TYPE]['edge_label_index'] = target_edge_index
        # pred_data = pred_data.to(self.device)
        #
        # y_hat = self.model(pred_data.x_dict, pred_data.edge_index_dict,
        #                  pred_data[self.TARGET_EDGE_TYPE]['edge_label_index'])

        # clone the data to avoid changing the original data
        pred_data = self.data.clone()
        pred_data.to(self.device)

        # build mask for test_word
        test_word_idx = self.word_node_idx_by_name[test_word]
        mask = torch.zeros(len(pred_data.x), dtype=torch.bool)
        mask[test_word_idx] = True
        mask = mask.to(self.device)

        # mask out the current word
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
        lang_idxs = [self.all_langs.index(lang) for lang in langs]
        mask = torch.zeros(len(pred_data.x), dtype=torch.bool)
        for lang_idx in lang_idxs:
            mask[pred_data.x[:, lang_idx + 3] == 1] = True
        mask = mask.to(self.device)

        # mask out the current language
        pred_data.x[mask] = 0

        node_loader = NeighborLoader(
            pred_data,
            num_neighbors=[-1],
            batch_size=self.config['train']['batch_size'],
            input_nodes=mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        )

        y_hats = []
        node_idxs = []
        for batch in node_loader:
            batch.to(self.device)
            y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)
            y_hat = y_hat[:batch.batch_size]
            y_hats.append(y_hat.detach().cpu())
            node_idxs.append(batch.n_id[:batch.batch_size].detach().cpu())

        y_hat = torch.cat(y_hats, dim=0)
        node_idxs = torch.cat(node_idxs, dim=0)

        link_probs = y_hat.sigmoid()
        assert (mask.sum() == len(y_hat))
        assert (mask.sum() == len(node_idxs))
        assert (len(node_idxs) == len(set(node_idxs)))  # assert that every idx in node_idxs appears exactly once

        # print all positive predictions
        self.print_human_readable_predictions(None, y_hat, link_probs, None, threshold, max_print_words=int(1e10),
                                              save_to_csv_path=f'data/6_results/predicted_sds_{langs}.csv',
                                              word_idx_map=node_idxs)
        print('\n')

    def init_model(self, data):
        self.model = Model(
            in_channels=data.num_features,
            hidden_channels=config['model']['num_hidden'],
            out_channels=data.y.size(1),  # config['model']['embedding_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            # num_relations=data.num_relations,
            # metadata=data.metadata(),
        )

        # self.model = GAT(in_channels=data.num_features,
        #             hidden_channels=self.config['model']['num_hidden'],
        #             out_channels=data.num_features, #num_classes
        #             heads=self.config['model']['num_heads'],
        #             num_layers=self.config['model']['num_layers'],
        #             dropout=self.config['model']['dropout'],
        #             dtype=torch.float32)

        self.model.to(self.device)

    def test(self, print_human_readable_predictions=True, threshold=None, eval_train_set=False,
             compute_ideal_threshold=True):
        print('\nValidation set performance:')
        _, _, ideal_threshold, _, _, _, _, _, _, _ = self.evaluate(
            self.data.val_mask, ['weights', 'probs'], print_human_readable_predictions, True, threshold,
            compute_ideal_threshold=compute_ideal_threshold)

        for lang in self.gt_langs:
            print(f'\nValidation set performance for {lang}:')
            lang_mask = self.data.x[:, self.all_langs.index(lang) + 3] == 1
            lang_mask = lang_mask.to(self.device)
            lang_mask &= self.data.val_mask.to(self.device)

            _, _, _, _, _, _, _, _, _, _ = self.evaluate(
                lang_mask, [], False, True, ideal_threshold if threshold is None else threshold,
                compute_ideal_threshold=False)

        print('\nTest set performance:')
        test_loss, test_auc, _, false_positives, false_negatives, test_precision, test_recall, acc, test_f1, f1_loss = self.evaluate(
            self.data.test_mask, [], print_human_readable_predictions, True,
            ideal_threshold if threshold is None else threshold, compute_ideal_threshold=False)
        # print(
        #     f"Test Loss: {test_loss:.3f}, Test AUC: {test_auc:.3f}, Test F1 Loss: {f1_loss:.3f}, ideal threshold: {ideal_threshold:.2f}, false positives: {false_positives}, false negatives: {false_negatives}, precision: {precision:.2f}, recall: {recall:.2f}, ACC: {acc:.2f}, F1: {f1:.2f}")
        if wandb.run is not None:
            wandb.log({"test loss": test_loss, "test AUC": test_auc, "test F1 loss": f1_loss,
                       "val ideal threshold": ideal_threshold,
                       "false positives": false_positives, "false negatives": false_negatives,
                       "precision": test_precision,
                       "recall": test_recall, "ACC": acc, "F1": test_f1})
            wandb.watch(self.model)

        if eval_train_set:
            print('\nTrain set performance:')
            self.evaluate(self.data.train_mask, [], print_human_readable_predictions, True,
                          ideal_threshold if threshold is None else threshold, compute_ideal_threshold=False)

        return test_loss, test_auc, ideal_threshold, test_f1, test_precision

    def split_dataset(self):
        # split = T.RandomLinkSplit(
        #     num_val=0.1,
        #     num_test=0.1,
        #     is_undirected=True,
        #     add_negative_train_samples=False,
        #     #disjoint_train_ratio=0.3,
        #     neg_sampling_ratio=0.0,  # 2.0
        #     edge_types=[self.TARGET_EDGE_TYPE],
        #     rev_edge_types=[('word', 'rev_has', 'semantic_domain')], # todo: fix unequal number of rev edges and forward edges for full graph
        # )
        # self.train_data, self.val_data, self.test_data = split(self.data)

        split = T.RandomNodeSplit(
            num_val=0.1,
            num_test=0.1,
        )
        self.data = split(self.data)
        print(self.data)
        return

        self.add_negative_edges_to_data_split(True, 'all')

        print("Validating split...")
        # assert that there is no overlap in the train, val, test sets
        train_set = self.convert_edge_index_to_set_of_tuples(self.train_data[self.TARGET_EDGE_TYPE].edge_label_index)
        val_set = self.convert_edge_index_to_set_of_tuples(self.val_data[self.TARGET_EDGE_TYPE].edge_label_index)
        test_set = self.convert_edge_index_to_set_of_tuples(self.test_data[self.TARGET_EDGE_TYPE].edge_label_index)
        union = train_set.union(val_set).union(test_set)
        assert len(union) == len(train_set) + len(val_set) + len(test_set)

        # assert that there are #words x #domains edges
        assert len(union) == len(self.sd_node_names) * len(self.word_node_names)

        print("Creating train loader...")
        self.train_loader = LinkNeighborLoader(
            # very slow because it cuts the graph into pieces instead of loading it once
            data=self.train_data.to(self.device),
            num_neighbors=[-1, -1],  # [-1, -1, -1]  # [20, 10],
            # neg_sampling_ratio=1.0,
            edge_label_index=(
                self.TARGET_EDGE_TYPE, self.train_data[self.TARGET_EDGE_TYPE].edge_label_index.to(self.device)),
            edge_label=self.train_data[self.TARGET_EDGE_TYPE].edge_label.to(self.device),
            batch_size=self.config['train']['batch_size'],  # number of edges per batch
            shuffle=True,
            subgraph_type='bidirectional',
        )

        # # put the missing nodes back (i.e., the source lang nodes)
        # train_data.x = x
        # val_data.x = x
        # test_data.x = x

        # print('Target data before splitting:', data)
        # print('Train data:', train_data)
        # print('Val data:', val_data)
        # print('Test data:', test_data)
        #
        # print('\n')
        # print('Train data edge_label and edge_label_index:', train_data[self.TARGET_EDGE_TYPE].edge_label, train_data[self.TARGET_EDGE_TYPE].edge_label_index)
        # print('Val data edge_label and edge_label_index:', val_data[self.TARGET_EDGE_TYPE].edge_label, val_data[self.TARGET_EDGE_TYPE].edge_label_index)
        # print('Test data edge_label and edge_label_index:', test_data[self.TARGET_EDGE_TYPE].edge_label, test_data[self.TARGET_EDGE_TYPE].edge_label_index)

        # # convert tensors to set of tuples and add them to the forbidden_neg_train_edges
        # val_label_edges = self.convert_edge_index_to_set_of_tuples(self.val_data[self.TARGET_EDGE_TYPE].edge_label_index)
        # test_label_edges = self.convert_edge_index_to_set_of_tuples(self.test_data[self.TARGET_EDGE_TYPE].edge_label_index)
        # self.forbidden_neg_train_edges = val_label_edges | test_label_edges

        # # add edge_index to train_data, val_data, test_data
        # train_data.edge_index = torch.cat([train_data.edge_index, alignment_edge_index], dim=1)
        # self.add_backward_edges(train_data)
        #
        # val_data.edge_index = torch.cat([val_data.edge_index, alignment_edge_index], dim=1)
        # self.add_backward_edges(val_data)
        #
        # test_data.edge_index = torch.cat([test_data.edge_index, alignment_edge_index], dim=1)
        # self.add_backward_edges(test_data)

        print('\n')
        print('Train data:', self.train_data, '\n')
        print('Val data:', self.val_data, '\n')
        print('Test data:', self.test_data, '\n')

    # def export_dataset(self):
    #     writer = Writer('data/sd_word_graph')
    #
    #     graph = dgl.heterograph(
    #         {edge_type: (edge_store.edge_index[0], edge_store.edge_index[1]) for edge_type, edge_store in
    #          zip(self.data.edge_types, self.data.edge_stores) if 'rev_' not in edge_type[1]},
    #     )
    #
    #     nlabels = graph.ndata['label']
    #     num_classes = len(self.node_labels_all)
    #
    #     writer.add_graph(name='SD Word Graph', graph=graph, nlabels=nlabels, num_nlabel_types=num_classes)
    #     writer.close()

    def add_hop_edges_to_graph(self, graph, max_hops=2):
        # add edges for 2-hops to the graph (number of conv layers)
        for node in tqdm(graph, desc=f'Adding {max_hops}-hop edges to graph'):
            visited = {node: 0}
            queue = deque([(node, 0)])
            while queue:
                current_node, depth = queue.popleft()
                if depth < max_hops:  # We only want to explore nodes at this max depth
                    for neighbor in graph[current_node]:
                        if neighbor not in visited:
                            visited[neighbor] = depth + 1
                            queue.append((neighbor, depth + 1))
                            if visited[neighbor] in range(2, max_hops + 1):
                                graph.add_edge(node, neighbor)

    def partition_graph(self):
        # solve n-color problem for multiple hops
        hop_graph = self.G.copy()

        # remove sd nodes
        hop_graph.remove_nodes_from(self.sd_node_names)

        # self.add_hop_edges_to_graph(hop_graph)

        print('Partitioning graph...')
        color_by_node = nx.greedy_color(hop_graph)
        print('Done partitioning graph')

        # create a list of bit masks (one for each color)
        for color in range(max(color_by_node.values()) + 1):
            mask = torch.zeros(hop_graph.number_of_nodes(), dtype=torch.bool)
            for node, node_color in color_by_node.items():
                if node_color == color:
                    node_idx = self.word_node_idx_by_name[node]
                    mask[node_idx] = True

            # # split mask into two masks of equal size (to reduce memory usage)
            # (mask1, mask2) = torch.split(mask, math.ceil(mask.size(0) / 2))
            # # fill up the masks with "False" values to bring them back to the original size
            # mask1 = torch.cat((mask1, torch.zeros(mask.size(0) - mask1.size(0), dtype=torch.bool)))
            # mask2 = torch.cat((torch.zeros(mask.size(0) - mask2.size(0), dtype=torch.bool), mask2))

            self.color_masks.append(mask.to(self.device))
            # self.color_masks.append(mask1.to(self.device))
            # self.color_masks.append(mask2.to(self.device))

        # assert that the masks are complete
        assert (sum([m.sum() for m in self.color_masks]).item() == len(self.color_masks[0]))

    def build_dataset(self, select_only_gt_langs=True):
        # path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
        # dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
        # self.data = dataset[0].to(self.device)
        #
        # # Add user node features for message passing:
        # self.data['user'].x = torch.eye(self.data['user'].num_nodes, device=self.device)
        # # del self.data['user'].num_nodes
        #
        # # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
        # self.data = T.ToUndirected()(self.data)
        # # del self.data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
        #
        # # Perform a link-level split into training, validation, and test edges:
        # self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
        #     num_val=0.1,
        #     num_test=0.1,
        #     neg_sampling_ratio=0.0,
        #     edge_types=[('user', 'rates', 'movie')],
        #     rev_edge_types=[('movie', 'rev_rates', 'user')],
        # )(self.data)
        # return

        # self.G = build_example_network()
        self.build_network()  # also contains languages without gt sds
        complete_graph = self.G
        if select_only_gt_langs:
            subgraph = self.G.subgraph([n for n in self.G.nodes if
                                        self.G.nodes[n]['lang'] in self.gt_langs.union({'semantic_domain'})])
            self.G, complete_graph = subgraph, self.G

        self.sd_node_names = [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "semantic_domain"]
        self.word_node_names = [node for node in self.G.nodes if self.G.nodes[node]["lang"] != "semantic_domain"]
        semantic_domain_nodes_set = set(self.sd_node_names)
        word_nodes_set = set(self.word_node_names)

        # Create dictionaries to map nodes to indices
        self.sd_node_idx_by_name = {node: i for i, node in enumerate(self.sd_node_names)}
        self.word_node_idx_by_name = {node: i for i, node in enumerate(self.word_node_names)}

        self.word_node_names_by_sd_name = {sd_node: self.G.neighbors(sd_node) for sd_node in self.sd_node_names}

        sd_edges = [(self.sd_node_idx_by_name[edge[0]], self.word_node_idx_by_name[edge[1]]) for edge in self.G.edges if
                    edge[0] in semantic_domain_nodes_set and edge[1] in word_nodes_set]
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
        print({lang: len(self.word_node_idxs_by_lang[lang]) for lang in self.word_node_idxs_by_lang})

        self.convert_empty_questions_by_lang_to_tensors()

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
        qid_feature = torch.zeros(len(self.word_node_names), len(self.sd_node_names))
        for node in tqdm(self.word_node_names, desc='Creating QID feature', total=len(self.word_node_names)):
            # set a 1 for each QID to which the word is connected
            for neighbor in self.G.neighbors(node):
                if neighbor in self.sd_node_names:
                    qid_feature[self.word_node_idx_by_name[node]][self.sd_node_idx_by_name[neighbor]] = 1

        # Create the QID count feature (count qids in qid_feature)
        qid_count_feature = torch.zeros(len(self.word_node_names), 1)
        for node in self.word_node_names:
            qid_count_feature[self.word_node_idx_by_name[node]] = torch.sum(
                qid_feature[self.word_node_idx_by_name[node]])

        # concatenate the features
        x = torch.cat((qid_count_feature, degree_feature, weighted_degree_feature, lang_feature, qid_feature), dim=1)

        # self.data = HeteroData()
        # self.data['semantic_domain'].x = torch.eye(len(semantic_domain_nodes), device=self.device)
        # self.data['word'].x = x # lang_feature

        target_edge_index = torch.tensor(sd_edges).transpose(0, 1)
        alignment_edge_index = torch.tensor(alignment_edges).transpose(0, 1)
        alignment_edge_weight = torch.tensor(
            [self.G.edges[(self.word_node_names[edge[0]], self.word_node_names[edge[1]])].get("weight",
                                                                                              0)
             for edge in alignment_edges]).type(torch.FloatTensor)

        # data['semantic_domain', 'has', 'semantic_domain'].edge_index = ...
        # self.data['word', 'aligns_with', 'word'].edge_index = alignment_edge_index
        self.data = Data(x=x, edge_index=alignment_edge_index, y=qid_feature, edge_weight=alignment_edge_weight)
        self.data = T.ToUndirected()(self.data)
        # del self.data['word', 'rev_has', 'semantic_domain'].edge_label  # remove "reverse" label
        self.data = self.data.to(self.device)
        print(self.data)
        # self.export_dataset()

        if not torch.cuda.is_available():  # hack to run this only on my laptop
            self.palette = sns.color_palette('pastel')
            self.palette += [self.palette[2]]
            self.palette = {lang: color for lang, color in zip(self.node_labels_all, self.palette)}
            node_colors = [self.palette[self.G.nodes[node]["lang"]]
                           # if "lang" in self.G.nodes[node] else self.palette[lang_by_node_idx[node]]
                           for node in self.G.nodes]
            node_sizes = [self.G.nodes[node]["size"] * 100 if "size" in self.G.nodes[node] else 100
                          for node in self.G.nodes]

            self.plot_subgraph(self.G, 'eng: dog')
            self.plot_subgraph(self.G, 'eng: water')
            self.plot_subgraph(self.G, 'eng: earth')
            self.plot_subgraph(complete_graph, 'deu: läutern')
            self.plot_subgraph(complete_graph, 'deu: wasser')
            self.plot_subgraph(complete_graph, 'deu: schnell')

            # plot neighborhoods
            self.plot_subgraph(self.G, random.choice(
                [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "ind"]))
            self.plot_subgraph(self.G, random.choice(
                [node for node in self.G.nodes if self.G.nodes[node]["lang"] == "fra"]))
            self.plot_subgraph(complete_graph, random.choice(
                [node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == "deu"]))
            self.plot_subgraph(complete_graph, random.choice(
                [node for node in complete_graph.nodes if complete_graph.nodes[node]["lang"] == "gej"]))

            # # False negatives
            # self.plot_subgraph(self.G, 'fra: contraire')  # Word, Opposite
            # self.plot_subgraph(self.G, 'fra: réservé')  # Speak little
            # self.plot_subgraph(self.G, 'fra: poursuivre')  # Obsessed
            # self.plot_subgraph(self.G, 'fra: crier')  # Sports
            # self.plot_subgraph(self.G, 'fra: laver')  # Care for hair
            #
            # # False positives
            # self.plot_subgraph(self.G, 'fra: femelle')  # Hope
            # self.plot_subgraph(self.G, 'fra: fondre')  # Crop failure?
            # self.plot_subgraph(self.G, 'fra: disette')  # Escape
            # self.plot_subgraph(self.G, 'fra: communion')  # Track an animal
            # self.plot_subgraph(self.G, 'fra: affaire')  # Group of things

            # plot_graph_with_labels(self.G, "Complete data", node_sizes, node_colors, edges, target_edges, self.palette,
            #                        node_name_by_index, 100)

        self.split_dataset()
        self.partition_graph()

        print("Creating train loaders...")
        self.train_loaders = [NeighborLoader(
            self.data,
            num_neighbors=[-1],  # performs better with two iterations of neighbor sampling, don't ask me why
            batch_size=self.config['train']['batch_size'],
            input_nodes=self.data.train_mask.to(self.device) & color_mask.to(self.device),
            shuffle=True,
            subgraph_type='bidirectional',
        ) for color_mask in self.color_masks if
            (self.data.train_mask.to(self.device) & color_mask.to(self.device)).sum() > 0]

        self.criterion = f1_loss_function
        # self.criterion = precision_loss_function

        ## These two loss functions lead to worse F1 scores
        # weight_by_class = torch.tensor([0.0] * self.data.y.size(1))
        # most_common_class_count = self.data.y.sum(dim=0).max()
        # for i in range(len(weight_by_class)):
        #     class_count = self.data.y[:, i].sum()
        #     if class_count > 0:
        #         weight_by_class[i] = most_common_class_count / class_count
        #     else:
        #         weight_by_class[i] = 10000.0
        # self.criterion = torch.nn.CrossEntropyLoss(weight=weight_by_class.to(self.device))

        # y = self.data.y # self.train_data[self.TARGET_EDGE_TYPE].edge_label
        # pos_weight = (y == 0.).sum() / y.sum()
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # self.criterion = FalsePN_Loss(pos_weight, precision_reward=10.0).to(self.device)

        return

        if 0 and not torch.cuda.is_available():  # hack to execute this only on my laptop
            g = self.convert_to_networkx(self.train_data)  # , n_sample=100)
            self.plot_graph_with_labels(g, "Train data, encoder (edge_index)", node_sizes, node_colors,
                                        self.G.edges,
                                        sd_edges, 100)
            g = self.convert_edge_label_index_to_networkx(self.train_data)
            self.plot_graph_with_labels(g, "Train data, decoder (edge_label_index)", node_sizes, node_colors,
                                        self.G.edges, sd_edges, 100)

            g = self.convert_to_networkx(self.val_data)
            self.plot_graph_with_labels(g, "Val data, encoder (edge_index)", node_sizes, node_colors,
                                        self.G.edges,
                                        sd_edges, 100)
            g = self.convert_edge_label_index_to_networkx(self.val_data)
            self.plot_graph_with_labels(g, "Val data, decoder (edge_label_index)", node_sizes, node_colors,
                                        self.G.edges,
                                        sd_edges, 100)

            g = self.convert_to_networkx(self.test_data)
            self.plot_graph_with_labels(g, "Test data, encoder (edge_index)", node_sizes, node_colors, self.G.edges,
                                        sd_edges, 100)
            g = self.convert_edge_label_index_to_networkx(self.test_data)
            self.plot_graph_with_labels(g, "Test data, decoder (edge_label_index)", node_sizes, node_colors,
                                        self.G.edges,
                                        sd_edges, 100)

    def sample_dataset(self):
        selected_nodes = torch.tensor(range(10), dtype=torch.long).to(self.device)

        # Extract the subgraph
        subgraph_data = subgraph(selected_nodes, self.data.edge_index, edge_attr=self.data.edge_weight)

        # Create a new Data object for the subgraph
        self.data = Data(
            x=subgraph_data.x,
            edge_index=subgraph_data.edge_index,
            edge_weight=subgraph_data.edge_attr,
            y=subgraph_data.y
        )

    def run_gnn_pipeline(self):
        # torch.manual_seed(1)  # fixed seed for reproducibility
        if 'cuda' in self.device:
            # assert that this GPU's memory is not used by anyone else
            assert torch.cuda.memory_allocated(self.device) == 0

        # self.build_dataset(False)
        # # self.sample_dataset()
        # self.load_model('data/3_models/model_F10.13?_th-0.1?_eng+fra+ind+yor_jumping-snowflake-104.bin') # model_AUC-1.00_th0.10_deu+eng+fra+gej+hin+ind+spa_gallant-water-2.bin')
        # threshold = 0.1
        # self.test(threshold=threshold) # This is not expressive because this contains the LRL languages, too.
        # # self.predict('eng: dog', threshold)
        # # self.predict('eng: swift', threshold)
        # # self.predict('eng: water', threshold)
        # #
        # # self.predict('deu: läutern', threshold)
        # # self.predict('deu: wasser', threshold)
        # # self.predict('deu: schnell', threshold)
        # # self.predict_for_languages(['deu', 'gej'], threshold)
        # self.predict_for_languages(['yor'], threshold)
        # return

        self.build_dataset()
        self.init_model(self.data)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['optimizer'][
            'learning_rate'], weight_decay=self.config['optimizer']['weight_decay'])
        assert (self.data.is_undirected())

        # ml wandb.init(project="Link prediction, Movie Lens dataset")
        self.train_link_predictor()

        # path_1 = 'data/3_models/model_proud-pond-143.bin'
        # self.load_model(path_1)
        path_1 = f"{self.config['train']['model_path']}_{wandb.run.name}.bin"
        self.save_model(path_1)
        test_loss, test_auc, ideal_threshold, test_f1, test_precision = self.test(eval_train_set=True, threshold=1.00,
                                                                                  compute_ideal_threshold=False)
        path_2 = path_1[
                 :-4] + f"_precision{test_precision:.2f}_F1{test_f1:.2f}_th{ideal_threshold:.2f}_{'+'.join(self.dc.target_langs)}.bin"
        self.save_model(path_2)
        os.remove(path_1)

        self.predict('eng: bright', 1.00)

        # load and test model again
        # self.load_model(path)
        # self.test(print_human_readable_predictions=False, threshold=ideal_threshold)

        # self.explain()


if __name__ == '__main__':
    print("GNN setup started")

    parser = argparse.ArgumentParser(description='Arguments for multi label classification training')

    parser.add_argument(
        '-config',
        help='Path to data self.config file',
        type=str, default='configs/gnn_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    wandb.init(
        # project="GCN node prediction (link semantic domains to target words), eng-fra-ind-hin-spa-gej-deu dataset",
        # project="GCN node prediction (link semantic domains to target words), eng-fra-ind-por-swa-spa-gej-deu dataset",
        # project="GCN node prediction (link semantic domains to target words), eng-fra-ind-yor dataset",
        project="GCN node prediction (link semantic domains to target words), 18 langs (nep) dataset",
        config={
            "learning_rate": config['optimizer']['learning_rate'],
            "epochs": config['train']['epochs'],
            "batch_size": config['train']['batch_size'],
            "dropout": config['model']['dropout'],
            "weight_decay": config['optimizer']['weight_decay'],
            "hidden_channels": config['model']['num_hidden'],
            "num_layers": config['model']['num_layers'],
            # "num_heads": config['model']['num_heads'],
        })

    # dc = LinkPredictionDictionaryCreator(
    #     ['bid-eng-web', 'bid-fra-fob', 'bid-ind', 'bid-hin', 'bid-spa', 'bid-gej', 'bid-deu'])
    # lp = LinkPredictor(dc, {'eng', 'fra', 'ind', 'hin', 'spa'}, config,
    # #                    'data/7_graphs/graph-all-qids-except-missing.cpickle')
    # #                    'data/7_graphs/graph-10-qids.cpickle')
    #                    'data/7_graphs/graph.cpickle')

    # dc = LinkPredictionDictionaryCreator(
    # ['bid-eng-web', 'bid-fra-fob', 'bid-ind', 'bid-por', 'bid-swa', 'bid-spa', 'bid-gej', 'bid-deu'])
    # lp = LinkPredictor(dc, {'eng', 'fra', 'ind', 'por', 'swa', 'spa'}, config,
    #                    #                   'data/7_graphs/graph-por.cpickle')
    #                    # 'data/7_graphs/graph_por_normalized-edge-weights_filtered-0.2_missing-questions.cpickle')
    #                     'data/7_graphs/graph_por_normalized-edge-weights_filtered-3-0.2_missing-questions.cpickle')

    # dc = LinkPredictionDictionaryCreator(
    # ['bid-eng-web', 'bid-fra-fob', 'bid-ind', 'bid-yor'])
    # lp = LinkPredictor(dc, {'eng', 'fra', 'ind'}, config,
    # 'data/7_graphs/graph-yor-all-qids.cpickle')
    # 'data/7_graphs/graph-yor-50-qids.cpickle')
    # 'data/7_graphs/graph-yor-normalized-edge-weights-filtered-0.2.cpickle')

    dc = LinkPredictionDictionaryCreator(['bid-eng-web', 'bid-fra-fob', 'bid-ind', 'bid-por', 'bid-swa', 'bid-spa',
                                          'bid-mya', 'bid-cmn', 'bid-hin', 'bid-mal', 'bid-nep', 'bid-urd', 'bid-pes',
                                          'bid-gej', 'bid-deu', 'bid-yor', 'bid-tpi', 'bid-meu'])
    lp = LinkPredictor(dc, {'eng', 'fra', 'ind', 'por', 'swa', 'spa', 'mya', 'cmn', 'hin', 'mal', 'arb', 'nep', 'urd',
                            'pes', 'gej', 'deu', 'yor', 'tpi', 'meu'}, config,
                       'data/7_graphs/graph-nep.cpickle')

    lp.run_gnn_pipeline()
