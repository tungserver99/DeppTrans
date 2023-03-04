import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from pytorch_models.common_pytorch_models import TextCNN, TorchFM, transModel # TorchFM doesn't include global bias

class DeepCoNN(nn.Module):
    def __init__(self, hyper_params):
        super(DeepCoNN, self).__init__()
        self.hyper_params = hyper_params
        
        word_vectors = load_obj(hyper_params['data_dir'] + '/word2vec')
        self.word2vec = nn.Embedding.from_pretrained(FloatTensor(word_vectors))
        self.word2vec.requires_grad = False # Not trainable

        self.user_conv = TextCNN(hyper_params)
        self.item_conv = TextCNN(hyper_params)

        self.final = nn.Sequential(
            nn.Linear(2 * hyper_params['latent_size'], hyper_params['latent_size']),
            nn.ReLU(),
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(hyper_params['latent_size'], 1)
        )

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 2) ]), requires_grad=True)
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 2) ]), requires_grad=True)
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]), requires_grad = True)

        self.fm = TorchFM(2 * hyper_params['latent_size'], 8)

        self.dropout = nn.Dropout(hyper_params['dropout'])
        self.relu = nn.ReLU()

    def forward(self, data):
        _, _, _, user_reviews, item_reviews, user_id, item_id = data

        final_shape = (user_id.shape[0])
        #print(user_id.shape)
        #print("final shape : " + final_shape)
        first_dim = user_id.shape[0]
        #print(item_id.shape)
        #print("first dim : " + first_dim)
        if len(user_id.shape) > 1:
            final_shape = (user_id.shape[0], user_id.shape[1])
            first_dim = user_id.shape[0] * user_id.shape[1]

        # For handling negatives
        user_reviews = user_reviews.view(first_dim, -1)
        item_reviews = item_reviews.view(first_dim, -1)
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        # Embed words
        user = self.word2vec(user_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        #print("User word embedding shape: " + user.shape)
        item = self.word2vec(item_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        #print("Item word embedding shape: " + item.shape)

        # Extract features
        user = self.user_conv(user)                      # [bsz x 32]
        #print("User feature extraction shape: " + user.shape)
        item = self.item_conv(item)                      # [bsz x 32]
        #print("Item feature extraction shape: " + item.shape)

        # Concatenate and get single score
        cat = torch.cat([ user, item ], dim = -1)
        
        # FM
        if self.hyper_params['model_type'] == 'deepconn':
            rating = self.global_bias + self.fm(cat)[:, 0]
            return rating.view(final_shape)

        # DeepCoNN ++
        rating = self.final(cat)[:, 0] # [bsz]
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)
        return (rating + user_bias + item_bias + self.global_bias).view(final_shape)

class DeepTrans(nn.Module):
    def __init__(self, hyper_params):
        super(DeepTrans, self).__init__()
        self.hyper_params = hyper_params
        
        word_vectors = load_obj(hyper_params['data_dir'] + '/word2vec')
        self.word2vec = nn.Embedding.from_pretrained(FloatTensor(word_vectors))
        self.word2vec.requires_grad = False # Not trainable

        self.user_conv = transModel(hyper_params)
        self.item_conv = transModel(hyper_params)

        self.final = nn.Sequential(
            nn.Linear(2 * hyper_params['latent_size'], hyper_params['latent_size']),
            nn.ReLU(),
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(hyper_params['latent_size'], 1)
        )

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 2) ]), requires_grad=True)
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 2) ]), requires_grad=True)
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]), requires_grad = True)

        self.fm = TorchFM(2 * hyper_params['latent_size'], 8)

        self.dropout = nn.Dropout(hyper_params['dropout'])
        self.relu = nn.ReLU()

    def forward(self, data):
        _, _, _, user_reviews, item_reviews, user_id, item_id = data

        final_shape = (user_id.shape[0])
        #print(user_id.shape)
        #print("final shape : " + str(final_shape))
        first_dim = user_id.shape[0]
        #print(item_id.shape)
        #print("first dim : " + str(first_dim))
        if len(user_id.shape) > 1:
            final_shape = (user_id.shape[0], user_id.shape[1])
            first_dim = user_id.shape[0] * user_id.shape[1]

        # For handling negatives
        user_reviews = user_reviews.view(first_dim, -1)
        item_reviews = item_reviews.view(first_dim, -1)
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        # Embed words
        user = self.word2vec(user_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        #print("User word embedding shape: ")
        #print(user.shape)
        item = self.word2vec(item_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        #print("Item word embedding shape: ")
        #print(item.shape)

        # Extract features
        user = self.user_conv(user)                      # [bsz x 32]
        #print("User feature extraction shape: ")
        #print(user.shape)
        item = self.item_conv(item)                      # [bsz x 32]
        #print("Item feature extraction shape: ")
        #print(item.shape)

        # Concatenate and get single score
        cat = torch.cat([ user, item ], dim = -1)
        
        # FM
        if self.hyper_params['model_type'] == 'deepconn' or self.hyper_params['model_type'] == 'deeptrans':
            rating = self.global_bias + self.fm(cat)[:, 0]
            return rating.view(final_shape)

        # DeepCoNN ++
        rating = self.final(cat)[:, 0] # [bsz]
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)
        return (rating + user_bias + item_bias + self.global_bias).view(final_shape)
