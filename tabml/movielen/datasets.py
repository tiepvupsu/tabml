import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split


class FactorizationMachineDataset(Dataset):
    def __init__(self, user_df, item_df, rating_df):
        self.user_df = user_df
        self.item_df = item_df
        self.rating_df = self.rating_df
        self.age_mapping = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
        self.gender_mapping = {"M": 0, "F": 1}
        self.occupation_mapping = {
            "other": 0,
            "academic/educator": 1,
            "artist": 2,
            "clerical/admin": 3,
            "college/grad student": 4,
            "customer service": 5,
            "doctor/health care": 6,
            "executive/managerial": 7,
            "farmer": 8,
            "homemaker": 9,
            "K-12 student": 10,
            "lawyer": 11,
            "programmer": 12,
            "retired": 13,
            "sales/marketing": 14,
            "scientist": 15,
            "self-employed": 16,
            "technician/engineer": 17,
            "tradesman/craftsman": 18,
            "unemployed": 19,
            "writer": 20,
        }
        self.genre_mapping = {
            "Action": 0,
            "Adventure": 1,
            "Animation": 3,
            "Children's": 4,
            "Comedy": 5,
            "Crime": 6,
            "Documentary": 7,
            "Drama": 8,
            "Fantasy": 9,
            "Film-Noir": 10,
            "Horror": 11,
            "Musical": 12,
            "Mystery": 13,
            "Romance": 14,
            "Sci-Fi": 15,
            "Thriller": 16,
            "War": 17,
            "Western": 18,
        }

    def __len__(self):
        return len(self.rating_df)

    def __getitem__(self, ind):
        """
        movie_id, movie_genre, user_id, user_gender, user_age, user_occupation
        """
        pass
