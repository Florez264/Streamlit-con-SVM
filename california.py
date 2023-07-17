import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

def mostrar_svm():
    st.header("SVM")
    
    # Cargar el dataset
    dataset = pd.read_csv('./data/california_housing.csv')

    # 1. Preparación de los datos
    X = dataset.drop(columns=["MedHouseVal"])
    y = dataset["MedHouseVal"]

    # 2. División de los datos
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Escalado de características
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Entrenamiento del modelo SVM
    svm = SVR(kernel='rbf')  # Puedes ajustar el kernel según tus necesidades
    svm.fit(X_train_scaled, y_train)

    # 5. Validación del modelo
    y_pred = svm.predict(X_val_scaled)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    st.write("RMSE:", rmse)

    y_pred = svm.predict(X_val_scaled)
    r2 = r2_score(y_val, y_pred)
    accuracy = r2 * 100  # predecimos la precisión
    st.write("Precisión:", accuracy)

    # 6. Ingreso de características para predicción
    st.subheader("Ingresar características de la vivienda:")
    house_features = []
    #Se crea una lista vacía para almacenar los valores ingresados por el usuario para cada caracteristica.
    for feature_name in X.columns: #Se itera sobre cada columna
        feature_values = pd.to_numeric(X[feature_name], errors='coerce') # convertir los valores de la columna en tipo numérico. 
        min_value = float(feature_values.min())
        max_value = float(feature_values.max())
        default_value = (min_value + max_value) / 2
        value = st.slider(feature_name, min_value, max_value, default_value)
        house_features.append(value)
    #Se escala  para asegurarse de que los valores estén en la misma escala que los datos de entrenamiento
    house_features_scaled = scaler.transform([house_features])
    predicted_value = svm.predict(house_features_scaled)
    st.write("Valor estimado de la vivienda:", predicted_value)

    # 7. Mostrar imagen condicionalmente
    if predicted_value <= 2.6000:
        st.image('./img/california.jpg')
    else:
        st.image('./img/lujocalifornia.jpg')

def main():
    st.sidebar.title("Menú")
    menu_options = ["SVM"]
    selected_option = st.sidebar.radio("", menu_options)
    
    if selected_option == "SVM":
        mostrar_svm()

if __name__ == "__main__":
    main()
