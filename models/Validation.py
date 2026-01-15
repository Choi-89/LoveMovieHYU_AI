import tensorflow as tf
import tensorflow_recommenders as tfrs

model_path = "C:\\capstone\\LoveMovieHYU-LMHYU_DipModel\\tfrs_model"

def validate_model(feel:str):
    try:
        model = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return ["모델을 불러올 수 없습니다."]
    
    _, recommendations = model(tf.constant([feel]))
    return [title.decode('utf-8') for title in recommendations[0, :10].numpy()]