import pytest
from HeartPredict.tests.test_app.main import app
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    """ Testa a página inicial """
    response = client.get('/')
    assert response.status_code == 200
    assert b'Resultados da Previsão' in response.data

def test_train_model_randomforest(client):
    """ Testa o treinamento do modelo RandomForest """
    response = client.post('/train', data={'modelo': 'randomforest'})
    assert response.status_code == 200
    assert b'Confusão' in response.data or b'Erro' in response.data

def test_train_model_logisticregression(client):
    """ Testa o treinamento do modelo LogisticRegression """
    response = client.post('/train', data={'modelo': 'logisticregression'})
    assert response.status_code == 200
    assert b'Confusão' in response.data or b'Erro' in response.data

def test_train_model_kneighbors(client):
    """ Testa o treinamento do modelo KNeighbors """
    response = client.post('/train', data={'modelo': 'kneighbors'})
    assert response.status_code == 200
    assert b'Confusão' in response.data or b'Erro' in response.data

def test_predict(client):
    """ Testa a previsão com dados de exemplo """
    # Treinar o modelo antes de testar a previsão
    client.post('/train', data={'modelo': 'randomforest'})

    response = client.post('/predict', data={
        'idade': '45',
        'sexo': 'masculino',
        'pressao_repouso': '120',
        'colesterol': '200',
        'acucar_sangue': 'falso',
        'freq_card_max': '150',
        'angina_induzida': 'não',
        'pico_antigo': '0.0',
        'inclina_pico_st': '1',
        'num_vasos_coloridos': '0',
        'thal': '3',
        'dor_peito': '0',
        'eletro_repouso': '0'
    })
    assert response.status_code == 200
    assert b'Probabilidade' in response.data or b'Predição' in response.data
