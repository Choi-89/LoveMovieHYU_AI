import pandas as pd

def add_dataset(user_id:int, movie_id:int, P:float, E:float, I:float):
    new_entry = pd.DataFrame([{
        "user_id": user_id,
        "movie_id": movie_id,
        "P": P,
        "E": E,
        "I": I
    }])
    file_name = "add_data.csv"
    df = pd.read_csv(file_name) if pd.io.common.file_exists(file_name) else pd.DataFrame()
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_name, index=False)

    print(df.head())