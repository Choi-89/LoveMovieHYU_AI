import pandas as pd

class Add_Movie:
    def __init__(self, user_id: int, movie_id: int, P: float, E: float, I: float):
        self.user_id = str(user_id)
        self.movie_id = str(movie_id)
        self.P = P
        self.E = E
        self.I = I

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "movie_id": self.movie_id,
            "P": self.P,
            "E": self.E,
            "I": self.I
        }


def add_dataset(Add_Movie: list):
    data_dict_list = [item.to_dict() for item in Add_Movie]
    new_entry = pd.DataFrame(data_dict_list)
    df = new_entry
    file_name = "add_data.csv"
    df.to_csv(file_name, index=False)

    print(df.head())