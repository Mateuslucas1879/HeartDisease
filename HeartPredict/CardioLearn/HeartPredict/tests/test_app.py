import pytest
from flask import Flask
import pandas as pd
from HeartPredict.tests.test_app.main import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_data():
    data = {
        'age': [63, 67, 67, 37],
        'sex': [1, 1, 1, 1],
        'cp': [1, 4, 4, 3],
        'trestbps': [145, 160, 120, 130],
        'chol': [233, 286, 229, 250],
        'fbs': [1, 0, 0, 0],
        'restecg': [2, 2, 2, 0],
        'thalach': [150, 108, 129, 187],
        'exang': [0, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5],
        'slope': [3, 2, 2, 3],
        'ca': [0, 3, 2, 0],
        'thal': [6, 3, 7, 3],
        'target': [1, 0, 0, 0]
    }
    return pd.DataFrame(data)


def test_preprocess_data(sample_data):
    df_encoded = preprocess_data(sample_data)
    assert 'cp_1' in df_encoded.columns
    assert 'restecg_2' in df_encoded.columns


def test_train_route(client, sample_data):
    app.config['df'] = preprocess_data(sample_data)

    response = client.post('/train', data={'modelo': 'randomforest'})
    assert response.status_code == 200
    assert b'Matriz de Confus' in response.data


def test_predict_route(client, sample_data):
    app.config['df'] = preprocess_data(sample_data)
    _, _, _, _, _, _, _, _ = train_model('randomforest')

    response = client.post('/predict', data={
        'idade': 45,
        'sexo': 'masculino',
        'pressao_repouso': 120,
        'colesterol': 200,
        'acucar_sangue': 'falso',
        'freq_card_max': 150,
        'angina_induzida': 'não',
        'pico_antigo': 0.0,
        'inclina_pico_st': 1,
        'num_vasos_coloridos': 0,
        'thal': 3,
        'dor_peito': 0,
        'eletro_repouso': 1
    })
    assert response.status_code == 200
    assert b'Probabilidade de' in response.data
