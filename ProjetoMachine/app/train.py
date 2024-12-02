import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

df = pd.read_csv('data/student-mat.csv')

X = df.drop(columns=['G3', 'school'])
y = df['G3']

columns_to_remove = [
    'paid', 'Mjob', 'famsize', 'address', 'freetime', 
    'studytime', 'traveltime', 'famsup', 'guardian', 'Fjob'
]
X = X.drop(columns=columns_to_remove)

numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X = remove_outliers(X, numeric_columns)
y = y[X.index]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#PIPELINES
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

#PARAMETROS GridSearchCV
param_grid_lr = {
    'regressor__fit_intercept': [True, False]
}

param_grid_dt = {
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 10, 20],
    'regressor__min_samples_leaf': [1, 5, 10]
}

#HIPERPARAMETROS E AVALIACAO MODELOS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='neg_mean_squared_error')
grid_search_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=5, scoring='neg_mean_squared_error')

grid_search_lr.fit(X_train, y_train)
grid_search_dt.fit(X_train, y_train)

best_model_lr = grid_search_lr.best_estimator_
best_model_dt = grid_search_dt.best_estimator_

y_pred_lr = best_model_lr.predict(X_test)
y_pred_dt = best_model_dt.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)

print("Linear Regression:")
print(f'Mean Squared Error (MSE): {mse_lr}')
print(f'Mean Absolute Error (MAE): {mae_lr}')
print(f'R2 Score: {r2_lr}')
print(f'Root Mean Squared Error (RMSE): {rmse_lr}')

print("\nDecision Tree Regressor:")
print(f'Mean Squared Error (MSE): {mse_dt}')
print(f'Mean Absolute Error (MAE): {mae_dt}')
print(f'R2 Score: {r2_dt}')
print(f'Root Mean Squared Error (RMSE): {rmse_dt}')

#ESCOLHE MELHOR MODELO
if rmse_lr < rmse_dt:
    best_model = best_model_lr
    print("Best model: Linear Regression")
else:
    best_model = best_model_dt
    print("Best model: Decision Tree Regressor")

model_path = os.path.join('models', 'best_regression_model.pkl')
joblib.dump(best_model, model_path)
print(f'Modelo salvo em: {model_path}')

#GRAFICOS DE MODELO
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, y_pred_dt, alpha=0.5, label='Decision Tree Regressor')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Scatter Plot das Previsões vs Valores Reais')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.legend()
plt.show()

X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
onehot_columns = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, onehot_columns])
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_transformed['G2'], y=y_test, label='Valores Reais')
sns.scatterplot(x=X_test_transformed['G2'], y=y_pred_lr, label='Previsões Linear Regression')
sns.scatterplot(x=X_test_transformed['G2'], y=y_pred_dt, label='Previsões Decision Tree')
plt.xlabel('G2')
plt.ylabel('G3')
plt.title('Relação entre G2 e G3')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_transformed['reason_home'], y=y_test, label='Valores Reais')
sns.scatterplot(x=X_test_transformed['reason_home'], y=y_pred_lr, label='Previsões Linear Regression')
sns.scatterplot(x=X_test_transformed['reason_home'], y=y_pred_dt, label='Previsões Decision Tree')
plt.xlabel('Reason Home')
plt.ylabel('G3')
plt.title('Relação entre Reason Home e G3')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_transformed['activities_yes'], y=y_test, label='Valores Reais')
sns.scatterplot(x=X_test_transformed['activities_yes'], y=y_pred_lr, label='Previsões Linear Regression')
sns.scatterplot(x=X_test_transformed['activities_yes'], y=y_pred_dt, label='Previsões Decision Tree')
plt.xlabel('Activities Yes')
plt.ylabel('G3')
plt.title('Relação entre Activities Yes e G3')
plt.legend()
plt.show()