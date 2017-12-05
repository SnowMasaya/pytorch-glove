# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from visualization.glove_visualize import GloveVisualize
from data.data_loader import DataLoader
from utils import read_file


class TestGloveVisualize(TestCase):
    def test_visualize(self):
        self.test_data_loader = DataLoader()
        self.test_japanese_wiki_data = '../data/raw/source_replay_twitter_data_sort.txt'
        test_word2index, test_index2word, test_window_data, \
        test_X_ik, test_weightinhg_dict = self.test_data_loader.load_data(
            file_name=self.test_japanese_wiki_data)  # noqa

        model_name = '../models/glove_model_40.pth'
        test_word2index.update({'<UNK>': len(test_word2index)})

        self.test_glove_visualize = GloveVisualize(model_name=model_name)
        self.test_glove_visualize.visualize(vocab=self.test_data_loader.vocab)

