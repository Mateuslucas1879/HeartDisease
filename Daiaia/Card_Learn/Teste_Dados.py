import requests

url = 'http://127.0.0.1:5000/predict'  # URL do servidor Flask

# Dados do paciente
data = {
    'idade': 25,
    'sexo': '0',  # Masculino
    'dor_peito': 1,  # Angina Típica
    'pressao_repouso': 110,
    'colesterol': 150,
    'acucar_sangue': '0',  # Não
    'eletro_repouso': 0,  # Normal
    'freq_card_max': 200,
    'angina_induzida': '0',  # Não
    'pico_antigo': 0.0,
    'inclina_pico_st': 2,
    'num_vasos_coloridos': 0,
    'thal': 3
}

response = requests.post(url, data=data)

if response.status_code == 200:
    print(response.text)
else:
    print(f"Failed to get prediction, status code: {response.status_code}")
