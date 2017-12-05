# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from utils import USE_CUDA
import torch.optim as optim
import torch
import numpy as np
from utils import prepare_word
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils import get_batch


class Trainer(object):

    def __init__(self,
                 embedding_size: int=50,
                 batch_size: int=256,
                 epoch: int=50,
                 model: object=None,
                 hidden_size: int=512,
                 decoder_learning_rate: float=5.0,
                 lr: float=0.0001,
                 rescheduled: bool=False,
                 fine_tune_model: str=''
                 ):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch = epoch
        if USE_CUDA is True and model is not None:
            model = model.cuda()
        self.model = model
        if model is not None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        self.hidden_size = hidden_size
        self.decoder_learning_rate = decoder_learning_rate
        self.lr = lr
        self.rescheduled = rescheduled
        self.fine_tune_model = fine_tune_model

    def train_method(self, train_data: list):
        losses = []

        for epoch in range(self.epoch):
            for i, batch in enumerate(get_batch(batch_size=self.batch_size,
                                                train_data=train_data)):
                inputs, targets, coocs, weights = zip(*batch)

                inputs = torch.cat(inputs)
                targets = torch.cat(targets)
                coocs = torch.cat(coocs)
                weights = torch.cat(weights)
                self.model.zero_grad()

                loss = self.model(inputs, targets, coocs, weights)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.data.tolist()[0])
            if epoch % 10 == 0:
                print('Epoch : %d, mean loss : %.02f' % (epoch, np.mean(losses)))
                self.__save_model_info(inputs, epoch, losses)
                torch.save(self.model, './../models/glove_model_{0}.pth'.format(epoch))
                losses = []
        self.writer.add_graph(self.model, loss)
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def __save_model_info(self, inputs, epoch, losses):
        x = vutils.make_grid(inputs, normalize=True, scale_each=True)
        self.writer.add_image('Image', x, epoch)
        self.writer.add_scalar('data/scalar1', np.mean(losses), epoch)

    def word_similarity(self,
                        target: list,
                        vocab: list,
                        word2index: dict,
                        top_rank: int=10
                        ):
        if USE_CUDA is True:
            target_V = self.model.prediction(prepare_word(target, word2index))
        else:
            target_V = self.model.prediction(prepare_word(target, word2index))
        similarities = []
        for i in range(len(vocab)):
            if vocab[i] == target:
                continue

            if USE_CUDA:
                vector = self.model.prediction(prepare_word(list(vocab)[i],
                                                            word2index))
            else:
                vector = self.model.prediction(prepare_word(list(vocab)[i],
                                                            word2index))
            consine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
            similarities.append([vocab[i], consine_sim])
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_rank]
