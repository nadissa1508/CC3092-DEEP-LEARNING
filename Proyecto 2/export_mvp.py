import nbformat as nbf
import json

file_path = '01_EDA_and_DataPrep.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells_to_add = [
    nbf.v4.new_markdown_cell('### Exportación de datos para el MVP (Componente 4)\nVamos a extraer 10 cuentas de prueba (5 normales y 5 de lavado) con sus probabilidades, scores de anomalía y pesos de atención precalculados para alimentar la aplicación de Streamlit.'),
    nbf.v4.new_code_cell('''import json

# Seleccionar 5 normales y 5 sospechosos
test_seqs = sequences[test_idx_B]
test_labels = labels[test_idx_B]
test_accounts = accounts[test_idx_B]

pos_idx = np.where(test_labels == 1)[0][:5]
neg_idx = np.where(test_labels == 0)[0][:5]
sample_idx = np.concatenate([pos_idx, neg_idx])

mvp_data = []

model_A.eval()
model_B.eval()

with torch.no_grad():
    for idx in sample_idx:
        seq = torch.FloatTensor(test_seqs[idx]).unsqueeze(0).to(device)
        
        # Score de Etapa A
        reconstructed, _ = model_A(seq)
        mse = torch.mean((reconstructed - seq)**2).item()
        
        # Probabilidad de Etapa B y Atención
        logits, attn = model_B(seq)
        prob = torch.sigmoid(logits).item()
        attn_weights = attn.squeeze().cpu().numpy().tolist()
        
        # Recuperar data original de transacciones para mostrar en la UI
        cuenta = test_accounts[idx]
        txs = df[df['SenderAccount'] == cuenta].sort_values('Timestamp').tail(MAX_SEQ_LEN)
        
        transacciones = []
        for _, row in txs.iterrows():
            transacciones.append({
                'Fecha': str(row['Timestamp']),
                'Monto': float(row['Amount']),
                'Hora': int(row['Hour'])
            })
            
        mvp_data.append({
            'Cuenta': cuenta,
            'Real_Lavado': bool(test_labels[idx]),
            'Score_Anomalia_EtapaA': mse,
            'Probabilidad_EtapaB': prob,
            'Atencion': attn_weights,
            'Transacciones': transacciones
        })

with open('mvp_data.json', 'w') as f:
    json.dump(mvp_data, f, indent=4)
print("Datos para MVP exportados a mvp_data.json")''')
]

nb['cells'].extend(cells_to_add)

with open(file_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

