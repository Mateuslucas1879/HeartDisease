import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# Carregar a base de dados
df = pd.read_csv('heart-disease.csv')

# Pré-processar dados
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['cp', 'restecg'])
    return df_encoded

df = preprocess_data(df)

def train_and_evaluate_logistic_regression():
    try:
        # Dados para treino do modelo
        X = df.drop("target", axis=1)
        y = df["target"]

        # Divisão em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Escalonar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Inicializar e treinar o modelo
        model = LogisticRegression(random_state=1, max_iter=1000)
        model.fit(X_train, y_train)

        # Fazer previsões
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades de classe 1 (doente)
        y_pred = (y_prob >= 0.5).astype(int)  # Converter probabilidades em classes

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_prob, squared=False)  # RMSE das probabilidades

        # Número de acertos e erros
        acertos = sum(y_test == y_pred)
        erros = len(y_test) - acertos
        total_predicoes = len(y_test)  # Número total de predições

        # Imprimir predições e resultados
        print("\nResultados para o modelo: Regressão Logística")
        print("Predições dos Pacientes:")
        for i in range(len(y_test)):
            print(f"Paciente {i+1}: Verdadeiro = {y_test.iloc[i]}, Predito = {y_pred[i]}, Probabilidade = {y_prob[i]:.4f}")

        print(f"\nNúmero de Acertos: {acertos}")
        print(f"Número de Erros: {erros}")
        print(f"Total de Predições: {total_predicoes}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Acurácia: {accuracy:.4f}")

    except ValueError as ve:
        print(f"Erro de Valor: {str(ve)}")
    except Exception as e:
        print("Ocorreu um erro ao processar a sua solicitação. Por favor, tente novamente.")
        print(str(e))

# Avaliar o modelo de Regressão Logística
train_and_evaluate_logistic_regression()
