import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from xgboost import XGBClassifier
from sklearn.datasets import make_classification

# Criando um conjunto de dados de exemplo
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir a função personalizada de scoring corretamente
def stability_score(y_true, y_pred, model, X_train, y_train):
    acc_train = accuracy_score(y_train, model.predict(X_train))  # Acurácia no treino
    acc_test = accuracy_score(y_true, y_pred)  # Acurácia no teste
    print(acc_test)
    if acc_train == 0:  # Evitar divisão por zero
        return 0
    return acc_test / acc_train  # Índice de estabilidade do modelo

# Criar scorer personalizado com `make_scorer`
custom_scorer = make_scorer(
    stability_score,  # Função correta
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False
)

# Definição do modelo e espaço de busca
model = XGBClassifier()
param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': np.linspace(0.01, 0.3, 5),
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    scoring=custom_scorer,  # Utilizando nossa métrica
    n_iter=20,  # Número de combinações testadas
    cv=3,  # Validação cruzada
    verbose=1,
    n_jobs=-1
)

# Executar busca
random_search.fit(X_train, y_train)

# Exibir melhores parâmetros
print("Melhores parâmetros:", random_search.best_params_)