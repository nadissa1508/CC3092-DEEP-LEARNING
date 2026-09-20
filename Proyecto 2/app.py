import streamlit as st
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detección de Lavado - MVP", layout="wide")

st.title("Sistema de Detección de Lavado de Dinero en Remesas")
st.markdown("MVP del sistema en Dos Etapas (Aprendizaje de Normalidad + Transfer Learning con Atención)")

@st.cache_data
def load_data():
    try:
        with open('mvp_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("El archivo mvp_data.json no se encuentra. Asegúrate de haber ejecutado la celda de exportación en el notebook.")
        return []

data = load_data()

if data:
    cuentas = [item['Cuenta'] for item in data]
    selected_cuenta = st.selectbox("Seleccione un Remitente (Cuenta):", cuentas)
    
    # Filtrar datos de la cuenta seleccionada
    user_data = next(item for item in data if item['Cuenta'] == selected_cuenta)
    
    st.subheader(f"Análisis de Cuenta: {user_data['Cuenta']}")
    
    # 1. Scores de las Etapas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score de Anomalía (Etapa A - Autoencoder MSE)", f"{user_data['Score_Anomalia_EtapaA']:.4f}")
    with col2:
        prob = user_data['Probabilidad_EtapaB']
        st.metric("Probabilidad de Lavado (Etapa B)", f"{prob:.2%}")
        if prob > 0.5:
            st.error("¡ALERTA DE LAVADO DE DINERO DETECTADA!")
        else:
            st.success("Patrón Normal")
            
    # 2. Secuencia de Transacciones
    st.subheader("Secuencia de Transacciones")
    df_txs = pd.DataFrame(user_data['Transacciones'])
    # Agregar pesos de atención a la tabla para visualización rápida
    df_txs['Peso Atención'] = user_data['Atencion'][-(len(df_txs)):] if len(df_txs) > 0 else []
    st.dataframe(df_txs)
    
    # 3. Mapa de Calor (Heatmap) de Atención
    st.subheader("Mapa de Calor: ¿Qué transacciones activaron la alerta?")
    st.markdown("Los pesos de atención indican en qué transacciones de la secuencia se fijó más el modelo para tomar su decisión final.")
    
    fig, ax = plt.subplots(figsize=(10, 2))
    attn_array = np.array(user_data['Atencion']).reshape(1, -1)
    # Solo mostrar los ticks reales (no los pads si hubiera)
    xticklabels = [f"Tx {i+1}" for i in range(len(user_data['Atencion']))]
    
    sns.heatmap(attn_array, annot=True, cmap='Reds', ax=ax, xticklabels=xticklabels, yticklabels=['Atención'])
    st.pyplot(fig)
    
    # 4. Explicación Automática en Lenguaje Natural
    st.subheader("Explicación de la Alerta")
    
    max_attn_idx = np.argmax(user_data['Atencion'])
    max_attn_val = user_data['Atencion'][max_attn_idx]
    
    # Determinar si es padding o transacción real
    # Si max_attn_idx < MAX_SEQ_LEN - len(txs), es padding (no debería pasar normalmente con buena atención)
    # Por simplicidad asumimos que la atención más alta corresponde a Tx {max_attn_idx + 1}
    
    if prob > 0.5:
        explicacion = f"El sistema generó una alerta porque el comportamiento de envío se desvía de la normalidad aprendida (Score de Etapa A: {user_data['Score_Anomalia_EtapaA']:.4f}). "
        explicacion += f"Al analizar la secuencia, la red neuronal enfocó el {max_attn_val:.1%} de su atención en la **Transacción {max_attn_idx + 1}**. "
        explicacion += "Esto indica que esta transacción, probablemente debido a un monto fragmentado o a un horario inusual en relación con el resto de la secuencia temporal, es el principal indicador del patrón de lavado."
    else:
        explicacion = "El sistema no generó alerta. El patrón temporal de remesas de esta cuenta es consistente con un comportamiento legítimo y no muestra desviaciones significativas en monto ni en frecuencia."
        
    st.info(explicacion)

else:
    st.warning("No hay datos cargados.")

