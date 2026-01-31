import tensorflow as tf
import pandas as pd
import json

model_path = "C:\\capstone\\LoveMovieHYU-LMHYU_DipModel\\tfrs_model"

def validate_model(user_id: int, P: float, E: float, I: float):
    try:
        model = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return ["모델을 불러올 수 없습니다."]
    
    input_query = {
        'user_id': tf.constant([str(user_id)], dtype=tf.string),
        'P': tf.constant([[P]]),
        'E': tf.constant([[E]]),
        'I': tf.constant([[I]])
    }

    scores, titles = model(input_query)

    recommended_movie_ids = [{"movie_id":str(title)} for title in titles[0, :3].numpy()]
    json_data = json.dumps(recommended_movie_ids, indent=4)

    print(f"🎬 추천된 영화 ID TOP 5: {recommended_movie_ids}")

    return json_data

if __name__ == "__main__":
    # Case 1: 기존 유저가 신날 때
    validate_model(user_id="22", P=0.35, E=0.35, I=0.35)