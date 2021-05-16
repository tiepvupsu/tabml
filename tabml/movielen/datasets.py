from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_MOVIES = 3883
NUM_USERS = 6040


class FactorizationMachineDataset(Dataset):
    def __init__(self, user_df, movie_df, rating_df):
        self.user_df = user_df
        self.movie_df = movie_df
        self.rating_df = rating_df
        self.age_vocab = Vocab("data/ml-1m/ages.txt")
        self.gender_vocab = Vocab("data/ml-1m/genders.txt")
        self.occupation_vocab = Vocab("data/ml-1m/occupations.txt")
        self.genre_vocab = Vocab("data/ml-1m/genres.txt")
        self.dims = [
            NUM_MOVIES,
            NUM_USERS,
            self.gender_vocab.len,
            self.age_vocab.len,
            self.occupation_vocab.len,
            self.genre_vocab.len,
        ]
        self.input_dim = sum(self.dims)
        self.rating_col_inds = {"user_ind": 0, "movie_ind": 1, "rating": 2}
        self.user_col_inds = {
            "user_ind": 0,
            "gender": 1,
            "age": 2,
            "occupation": 3,
            "zip-code": 4,
        }
        self.movie_col_inds = {"movie_ind": 0, "title": 1, "genres": 2}
        # TODO (add zipcode and timestamp)

    def __len__(self):
        return len(self.rating_df)

    def __getitem__(self, ind):
        """
        movie_id, movie_genre, user_id, user_gender, user_age, user_occupation
        """
        user_ind = self.rating_df[self.rating_col_inds["user_ind"]].iloc[ind] - 1
        assert user_ind < NUM_USERS
        movie_id = self.rating_df[self.rating_col_inds["movie_ind"]].iloc[ind]
        movie_true_ind = self.find_movie_index(movie_id)
        assert movie_true_ind < NUM_MOVIES
        rating = self.rating_df[self.rating_col_inds["rating"]].iloc[ind]
        gender_ind = self.gender_vocab.find_index(
            self.user_df[self.user_col_inds["gender"]][user_ind]
        )
        assert gender_ind < 2
        age_ind = self.age_vocab.find_index(
            str(self.user_df[self.user_col_inds["age"]][user_ind])
        )
        assert age_ind < self.age_vocab.len
        # occupation is given as index already
        occupation_ind = self.user_df[self.user_col_inds["occupation"]][user_ind]
        assert occupation_ind < self.occupation_vocab.len
        genre_inds = []
        genres = self.movie_df[self.movie_col_inds["genres"]][movie_true_ind].split("|")
        if genres:
            genre_inds = [self.genre_vocab.find_index(genre) for genre in genres]
        inputs = self.gen_multihot_input_tensor(
            [
                [movie_true_ind],
                [user_ind],
                [gender_ind],
                [age_ind],
                [occupation_ind],
                genre_inds,
            ]
        )
        return inputs, rating

    def find_movie_index(self, movie_id):
        movie_inds = self.movie_df[0].tolist()
        return movie_inds.index(movie_id)

    def gen_multihot_input_tensor(self, list_inds: List[List[int]]):
        """Generates a 1-d tensor as input of factorization machine model.

        Args:
            list_inds: a list of list of indices for each field defined in self.dims

        Example:
            If self.dims = [3, 5] and list_inds = [[2], [1, 3]] then return a tensor
            with values: [0, 0, 1, 0, 1, 0, 1, 0].
        """
        assert len(list_inds) == len(
            self.dims
        ), f"len(list_inds) ({len(list_inds)})  != len(self.dims) ({len(self.dims)})."

        offset = 0
        sparse = torch.zeros((self.input_dim), dtype=torch.float)
        one_inds = []
        for i, inds in enumerate(list_inds):
            assert not inds or all([ind < self.dims[i] for ind in inds])
            one_inds.extend([offset + ind for ind in inds])
            offset += self.dims[i]
        indices = torch.LongTensor([one_inds])
        values = torch.ones_like(torch.tensor((len(one_inds),)), dtype=torch.float)
        sparse[indices] = values
        return sparse


class Vocab:
    def __init__(self, vocab_file: str):
        """
        Args:
            vocab_file: path to file containing dictionary, each line is a word.
        """
        self.vocab_file = vocab_file
        self.vocab_list = self.read_vocab_file()
        self.len = len(self.vocab_list)

    def read_vocab_file(self):
        """Returns a list of words."""
        with open(self.vocab_file) as f:
            words = f.read().splitlines()  # avoid newline at the end
        return words

    def find_index(self, word):
        return self.vocab_list.index(word)


def get_ml_1m_dataset():
    rating_df = pd.read_csv("data/ml-1m/ratings.dat", header=None, delimiter="::")
    user_df = pd.read_csv("data/ml-1m/users.dat", header=None, delimiter="::")
    movie_df = pd.read_csv("data/ml-1m/movies.dat", header=None, delimiter="::")
    rating_df_train = rating_df.sample(frac=0.8, random_state=200)
    rating_df_val = rating_df.drop(rating_df_train.index)

    return (
        FactorizationMachineDataset(user_df, movie_df, rating_df_train),
        FactorizationMachineDataset(user_df, movie_df, rating_df_val),
    )
