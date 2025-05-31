from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import os
import traceback

app = Flask(__name__)

# Carregar a base de dados
df = pd.read_csv('heart-disease.csv')


# Pré-processar dados
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['cp', 'restecg'])
    return df_encoded


df = preprocess_data(df)

# Escalonador global para consistência entre treinamento e predição
scaler = StandardScaler()


def train_model(modelo):
    try:
        # Dados para treino do modelo
        X = df.drop("target", axis=1)
        y = df["target"]

        # Divisão em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Escalonar os dados
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Escolher o modelo
        if modelo == 'randomforest':
            model = RandomForestClassifier(random_state=1)
        elif modelo == 'logisticregression':
            model = LogisticRegression(random_state=1, max_iter=1000)
        elif modelo == "kneighbors":
            model = KNeighborsClassifier()
        else:
            return None, None, None, None, None, None, None, "Modelo inválido. Por favor, escolha 'randomforest', 'logisticregression' ou 'kneighbors'."

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas
        accuracy = (y_pred == y_test).mean()
        classification_rep = classification_report(y_test, y_pred, target_names=["Não Doente", "Doente"])

        if modelo == 'logisticregression':
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            confusion_matrix_result = None  # Deixar a matriz de confusão em branco
        else:
            rmse = None
            confusion_matrix_result = confusion_matrix(y_test, y_pred).tolist()

        # Número de acertos e erros para todos os modelos
        acertos = sum(y_test == y_pred)
        erros = len(y_test) - acertos

        return model, confusion_matrix_result, acertos, erros, accuracy, rmse, classification_rep, None

    except ValueError as ve:
        return None, None, None, None, None, None, None, f"Erro de Valor: {str(ve)}"
    except Exception as e:
        traceback.print_exc()  # Imprime o rastro de pilha da exceção
        return None, None, None, None, None, None, None, "Ocorreu um erro ao processar a sua solicitação. Por favor, tente novamente."


@app.route('/')
def home():
    confusion_matrix = app.config.get('confusion_matrix')
    acertos = app.config.get('acertos')
    erros = app.config.get('erros')
    accuracy = app.config.get('accuracy')
    rmse = app.config.get('rmse')
    classification_rep = app.config.get('classification_rep')
    return render_template('index.html', confusion_matrix=confusion_matrix, acertos=acertos, erros=erros,
                           accuracy=accuracy, rmse=rmse, classification_rep=classification_rep)


@app.route('/train', methods=['POST'])
def train():
    try:
        modelo = request.form['modelo'].lower()
        model, confusion_matrix_result, acertos, erros, accuracy, rmse, classification_rep, error_message = train_model(modelo)

        if error_message:
            return error_message

        # Salvar o modelo treinado e resultados
        app.config['model'] = model
        app.config['model_type'] = modelo  # Adicione essa linha para armazenar o tipo de modelo
        app.config['confusion_matrix'] = confusion_matrix_result
        app.config['acertos'] = acertos
        app.config['erros'] = erros
        app.config['accuracy'] = accuracy
        app.config['rmse'] = rmse
        app.config['classification_rep'] = classification_rep

        return render_template('index.html', confusion_matrix=confusion_matrix_result, acertos=acertos, erros=erros, accuracy=accuracy, rmse=rmse, classification_rep=classification_rep)
    except Exception as e:
        traceback.print_exc()  # Imprime o rastro de pilha da exceção
        return "Ocorreu um erro ao processar a sua solicitação. Por favor, tente novamente."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = app.config.get('model')
        model_type = app.config.get('model_type')

        if model is None:
            return "Modelo não treinado. Por favor, treine o modelo antes de fazer previsões."

        # Obter os dados do formulário
        data = {
            'age': [int(request.form['idade'])],
            'sex': [0 if request.form['sexo'].lower() == 'masculino' else 1],
            'trestbps': [int(request.form['pressao_repouso'])],
            'chol': [int(request.form['colesterol'])],
            'fbs': [0 if request.form['acucar_sangue'].lower() == 'verdadeiro' else 1],
            'thalach': [int(request.form['freq_card_max'])],
            'exang': [0 if request.form['angina_induzida'].lower() == 'sim' else 1],
            'oldpeak': [float(request.form['pico_antigo'])],
            'slope': [int(request.form['inclina_pico_st'])],
            'ca': [int(request.form['num_vasos_coloridos'])],
            'thal': [int(request.form['thal'])]
        }

        # Incluir codificação one-hot para as variáveis categóricas
        cp = [0, 0, 0, 0]  # Ajuste para 4 categorias
        restecg = [0, 0, 0]  # Mantém como 3 categorias

        try:
            cp_index = int(request.form['dor_peito'])
            if 1 <= cp_index <= len(cp):
                cp[cp_index - 1] = 1  # Ajuste para indexar corretamente
            else:
                return "Valor inválido para 'dor_peito'. Deve estar entre 1 e 4."

            restecg_index = int(request.form['eletro_repouso'])
            if 0 <= restecg_index < len(restecg):
                restecg[restecg_index] = 1
            else:
                return "Valor inválido para 'eletro_repouso'. Deve estar entre 0 e 2."

        except ValueError:
            return "Valor inválido enviado para o formulário."

        data.update({
            'cp_0': cp[0],
            'cp_1': cp[1],
            'cp_2': cp[2],
            'cp_3': cp[3],
            'restecg_0': restecg[0],
            'restecg_1': restecg[1],
            'restecg_2': restecg[2],
        })

        # Preparar os dados do usuário para previsão
        user_data = pd.DataFrame(data)

        # Escalonar os dados do usuário
        user_data_scaled = scaler.transform(user_data)

        # Prever o resultado
        if model_type == 'logisticregression':
            user_prob = model.predict_proba(user_data_scaled)[0]  # Probabilidades de cada classe
        else:
            user_prob = [model.predict(user_data_scaled)[0]]  # Previsão da classe

        # Atribuir valor 'None' para rmse se não estiver disponível
        rmse = app.config.get('rmse')
        if rmse is None:
            rmse = 'N/A'

        return render_template('prediction_results.html', user_prob=user_prob, model_type=model_type, rmse=rmse)

    except Exception as e:
        traceback.print_exc()  # Imprime o rastro de pilha da exceção
        return "Ocorreu um erro ao processar a sua solicitação. Por favor, tente novamente."



if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
    app.run(debug=True)