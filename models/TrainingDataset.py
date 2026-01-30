import pandas as pd
import numpy as np
import random

max_movie = 8459
max_user = 50

# 장르별 바이오리듬 값 범위
GENRE_MAP = {
    '액션':       {'P': [ 0.4, 1.0], 'E': [-0.4, 0.6], 'I': [-0.8, 0.2]},
    '모험':       {'P': [ 0.3, 0.9], 'E': [-0.2, 0.8], 'I': [-0.6, 0.4]},
    '서부':       {'P': [ 0.3, 0.9], 'E': [-0.4, 0.4], 'I': [-0.4, 0.5]},
    '전쟁':       {'P': [ 0.6, 1.0], 'E': [-0.6, 0.4], 'I': [ -0.2, 0.7]},
    '로맨스':     {'P': [-0.8, 0.2], 'E': [ 0.2, 1.0], 'I': [-0.6, 0.3]},
    '가족':       {'P': [-0.9, 0.1], 'E': [ 0.4, 1.0], 'I': [-0.8, 0.2]},
    '음악':       {'P': [-0.5, 0.6], 'E': [ 0.3, 1.0], 'I': [-0.4, 0.5]},
    '드라마':     {'P': [0.9, 0.1], 'E': [ -0.2, 1.0], 'I': [ 0.0, 0.8]},
    'TV 영화':    {'P': [-0.7, 0.3], 'E': [ -0.2, 0.7], 'I': [-0.7, 0.3]},
    '다큐멘터리': {'P': [-1.0, -0.2], 'E': [-0.6, 0.4], 'I': [ 0.5, 1.0]},
    '역사':       {'P': [-0.6, 0.4], 'E': [-0.5, 0.5], 'I': [ 0.5, 1.0]},
    '미스터리':   {'P': [-0.5, 0.5], 'E': [-0.8, 0.0], 'I': [ 0.3, 0.8]},
    '범죄':       {'P': [ 0.1, 0.8], 'E': [-0.9, -0.1], 'I': [ 0.3, 0.9]},
    '스릴러':     {'P': [ 0.4, 0.9], 'E': [-0.8, 0.1], 'I': [ 0.3, 1.0]},
    'SF':         {'P': [ 0.1, 0.9], 'E': [-0.8, -0.1], 'I': [ 0.2, 0.9]},
    '공포':       {'P': [ 0.6, 1.0], 'E': [ -0.8, 0.1], 'I': [-0.6, 0.4]},
    '자극':       {'P': [ 0.7, 1.0], 'E': [ 0.5, 1.0], 'I': [-1.0, -0.2]},
    '코미디':     {'P': [-0.3, 0.6], 'E': [ 0.4, 1.0], 'I': [-0.8, -0.4]},
    '판타지':     {'P': [ 0.2, 0.8], 'E': [ 0.2, 0.9], 'I': [ 0.0, 0.6]},
    '애니메이션': {'P': [-0.6, 0.5], 'E': [ 0.2, 0.9], 'I': [-0.8, -0.3]},
    '집중':       {'P': [-0.8, -0.3], 'E': [-0.5, 0.2], 'I': [ 0.7, 1.0]},
}

def make_training_dataset():
    df = pd.read_csv('movies.csv')

    p_min = {k: v['P'][0] for k, v in GENRE_MAP.items()}
    p_max = {k: v['P'][1] for k, v in GENRE_MAP.items()}
    e_min = {k: v['E'][0] for k, v in GENRE_MAP.items()}
    e_max = {k: v['E'][1] for k, v in GENRE_MAP.items()}
    i_min = {k: v['I'][0] for k, v in GENRE_MAP.items()}
    i_max = {k: v['I'][1] for k, v in GENRE_MAP.items()}
    
    df['P_min'] = df['genres'].map(p_min)
    df['P_max'] = df['genres'].map(p_max)
    df['E_min'] = df['genres'].map(e_min)
    df['E_max'] = df['genres'].map(e_max)
    df['I_min'] = df['genres'].map(i_min)
    df['I_max'] = df['genres'].map(i_max)
    df.fillna(-0.2, inplace=True)

    df['user_id'] = random.choices(range(max_user), k=len(df))
    df['P'] = np.random.uniform(df['P_min'], df['P_max']).round(4)
    df['E'] = np.random.uniform(df['E_min'], df['E_max']).round(4)
    df['I'] = np.random.uniform(df['I_min'], df['I_max']).round(4)

    df_final = df.drop(columns=['P_min', 'P_max', 'E_min', 'E_max', 'I_min', 'I_max'])

    df_final.to_csv("./emotional_dataset.csv", index=False)
    print(df_final.head())

if __name__ == "__main__":
    make_training_dataset()