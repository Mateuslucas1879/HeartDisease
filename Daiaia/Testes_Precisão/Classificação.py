import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Carregar a base de dados
df = pd.read_csv('heart-disease.csv')

# Pré-processar dados
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['cp', 'restecg'])
    return df_encoded

df = preprocess_data(df)

def train_and_evaluate_model(modelo):
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

        # Escolher o modelo
        if modelo == 'kneighbors':
            model = KNeighborsClassifier()
        elif modelo == 'randomforest':
            model = RandomForestClassifier(random_state=1)
        else:
            print("Modelo inválido. Por favor, escolha 'kneighbors' ou 'randomforest'.")
            return

        model.fit(X_train, y_train)

        # Fazer previsões
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades de classe 1 (doente)
        y_pred = (y_prob >= 0.5).astype(int)  # Converter probabilidades em classes

        # Calcular métricas
        confusion_matrix_result = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(y_test, y_pred, target_names=["Não Doente", "Doente"])
        auc_score = roc_auc_score(y_test, y_prob)

        # Número de acertos e erros
        acertos = sum(y_test == y_pred)
        erros = len(y_test) - acertos
        total_predicoes = len(y_test)  # Número total de predições

        # Imprimir predições e resultados
        print(f"\nResultados para o modelo: {modelo}")
        print("Predições dos Pacientes:")
        for i in range(len(y_test)):
            print(f"Paciente {i+1}: Verdadeiro = {y_test.iloc[i]}, Predito = {y_pred[i]}, Probabilidade = {y_prob[i]:.4f}")

        print(f"\nNúmero de Acertos: {acertos}")
        print(f"Número de Erros: {erros}")
        print(f"Matriz de Confusão: {confusion_matrix_result}")
        print(f"Total de Predições: {total_predicoes}")
        print(f"AUC: {auc_score:.4f}")

        print("\nRelatório de Classificação:")
        print(class_report)

        return model, confusion_matrix_result, acertos, erros, auc_score, total_predicoes

    except ValueError as ve:
        print(f"Erro de Valor: {str(ve)}")
    except Exception as e:
        print("Ocorreu um erro ao processar a sua solicitação. Por favor, tente novamente.")
        print(str(e))

# Treinar e avaliar os modelos
modelos = ['kneighbors', 'randomforest']
for modelo in modelos:
    train_and_evaluate_model(modelo)
