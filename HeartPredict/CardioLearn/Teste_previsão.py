import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # Codificação one-hot para 'cp' e 'restecg'
    df_encoded = pd.get_dummies(df, columns=['cp', 'restecg'], drop_first=True)
    return df_encoded


def train_and_evaluate_model(model_name, df):
    try:
        print(f"\nTreinando e avaliando o modelo: {model_name}")

        # Dados para treino do modelo
        X = df.drop("target", axis=1)
        y = df["target"]

        # Divisão em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        print("Divisão dos dados completa.")

        # Escalonar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Escalonamento dos dados completo.")

        # Escolher o modelo
        if model_name == 'randomforest':
            model = RandomForestClassifier(random_state=1)
        elif model_name == 'logisticregression':
            model = LogisticRegression(random_state=1, max_iter=1000)
        elif model_name == 'kneighbors':
            model = KNeighborsClassifier()
        else:
            print("Modelo inválido.")
            return

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Treinamento do modelo completo.")

        # Calcular métricas
        accuracy = (y_pred == y_test).mean()
        print(f"Acurácia: {accuracy:.4f}")

        classification_rep = classification_report(y_test, y_pred, target_names=["Não Doente", "Doente"])
        print("Relatório de Classificação:")
        print(classification_rep)

        if model_name == 'logisticregression':
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            print(f"RMSE: {rmse:.4f}")
        else:
            rmse = None

        confusion_matrix_result = confusion_matrix(y_test, y_pred).tolist()
        print("Matriz de Confusão:")
        print(confusion_matrix_result)

    except ValueError as ve:
        print(f"Erro de Valor: {str(ve)}")
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")


def test_predict(df):
    try:
        # Dados de exemplo
        sample_data = {
            'age': [45],
            'sex': [0],
            'cp': [0, 0, 0, 1],  # Codificação one-hot para 'cp'
            'trestbps': [120],
            'chol': [200],
            'fbs': [0],
            'restecg': [0, 1, 0],  # Codificação one-hot para 'restecg'
            'thalach': [150],
            'exang': [0],
            'oldpeak': [0.0],
            'slope': [1],
            'ca': [0],
            'thal': [3]
        }

        user_data = pd.DataFrame(sample_data)
        print("\nDados de exemplo para previsão:")
        print(user_data)

        # Pré-processar dados
        df_encoded = preprocess_data(df)
        X = df_encoded.drop("target", axis=1)
        y = df_encoded["target"]

        # Treinar o modelo com o RandomForest para o exemplo
        model = RandomForestClassifier(random_state=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # Pré-processar os dados de usuário
        user_data_encoded = preprocess_data(user_data)

        # Imprimir colunas esperadas e colunas no dados do usuário
        print(f"Colunas esperadas: {X.columns}")
        print(f"Colunas no dados do usuário: {user_data_encoded.columns}")

        # Garantir que todas as colunas presentes no treinamento estejam na previsão
        missing_cols = set(X.columns) - set(user_data_encoded.columns)
        for col in missing_cols:
            user_data_encoded[col] = 0

        # Reordenar as colunas para garantir a correspondência
        user_data_encoded = user_data_encoded[X.columns]
        user_data_scaled = scaler.transform(user_data_encoded)

        print("\nDados do usuário após escalonamento:")
        print(user_data_scaled)

        # Fazer previsões
        user_prob = model.predict_proba(user_data_scaled)
        user_pred = model.predict(user_data_scaled)

        print("Probabilidade de Ataque Cardíaco:")
        print(user_prob)

        print("Predição Binária:")
        print(user_pred)

    except ValueError as ve:
        print(f"Erro de Valor: {str(ve)}")
    except Exception as e:
        print(f"\nOcorreu um erro na previsão: {str(e)}")


if __name__ == '__main__':
    # Carregar a base de dados
    df = pd.read_csv('heart-disease.csv')
    df = preprocess_data(df)

    # Testar diferentes modelos
    models = ['randomforest', 'logisticregression', 'kneighbors']
    for model_name in models:
        train_and_evaluate_model(model_name, df)

    # Testar a previsão com dados de exemplo
    test_predict(df)

