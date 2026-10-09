import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuração da página
st.set_page_config(page_title="Kovan LATAM - Inteligência Dinâmica de Churn", layout="wide", page_icon="📊")

@st.cache_data
def carregar_dados_dinamicos():
    excel_file = 'datasets_case_modulo2_5yrs.xlsx'
    todas_as_abas = pd.read_excel(excel_file, sheet_name=None)
    
    df_raw, df_clients, df_crm = None, None, None
    for nome_aba, df_aba in todas_as_abas.items():
        col = df_aba.columns.tolist()
        if 'billing_dt' in col and 'total_value_usd' in col: df_raw = df_aba.copy()
        elif 'tempo_como_cliente' in col and 'canal_aquisicao' in col: df_clients = df_aba.copy()
        elif 'contatos_realizados' in col and 'oportunidades_abertas' in col: df_crm = df_aba.copy()

    df_raw['billing_dt'] = pd.to_datetime(df_raw['billing_dt'])
    df_clients = df_clients[['account_id', 'segment', 'industry', 'canal_aquisicao']].drop_duplicates()
    
    crm_agg = df_crm.groupby('account_id').agg(
        contatos_realizados=('contatos_realizados', 'sum'),
        oportunidades_abertas=('oportunidades_abertas', 'sum'),
        oportunidades_ganhas=('oportunidades_ganhas', 'sum'),
        dias_sem_contato=('dias_sem_contato', 'max')
    ).reset_index()

    CUTOFF_DATE = pd.to_datetime('2024-03-31') 
    df_features_period = df_raw[df_raw['billing_dt'] <= CUTOFF_DATE].copy()

    # 1. REMOVER CLIENTES DE COMPRA ÚNICA
    order_counts = df_features_period.groupby('account_id')['invoice_no'].nunique()
    valid_accounts = order_counts[order_counts > 1].index
    df_features_period = df_features_period[df_features_period['account_id'].isin(valid_accounts)]

    # 2. RFM e CADÊNCIA
    rfm_div = df_features_period.groupby('account_id').agg(
        recency_date=('billing_dt', 'max'),
        frequency_orders=('invoice_no', 'nunique'),
        monetary_total=('total_value_usd', 'sum')
    ).reset_index()
    rfm_div['recency_days'] = (CUTOFF_DATE - rfm_div['recency_date']).dt.days
    rfm_div.drop(columns=['recency_date'], inplace=True)

    purchase_dates = df_features_period[['account_id', 'billing_dt']].drop_duplicates().sort_values(by=['account_id', 'billing_dt'])
    purchase_dates['days_since_previous'] = purchase_dates.groupby('account_id')['billing_dt'].diff().dt.days
    cadence = purchase_dates.groupby('account_id')['days_since_previous'].agg(['mean', 'std']).reset_index()
    cadence.columns = ['account_id', 'avg_days_between_purchases', 'std_days_between_purchases']

    # 3. RÉGUA DINÂMICA
    # Limite = 2x a média de dias, com piso mínimo de 90 dias
    cadence['limite_dinamico_dias'] = (cadence['avg_days_between_purchases'] * 2).clip(lower=90)

    t_6m = CUTOFF_DATE - pd.DateOffset(months=6)
    rev_6m = df_features_period[df_features_period['billing_dt'] >= t_6m].groupby('account_id')['total_value_usd'].sum().reset_index().rename(columns={'total_value_usd': 'revenue_6m'})

    all_clients = pd.DataFrame({'account_id': df_features_period['account_id'].unique()})
    df_model = all_clients.merge(rfm_div, on='account_id', how='left')
    df_model = df_model.merge(cadence, on='account_id', how='left')
    df_model = df_model.merge(rev_6m, on='account_id', how='left')
    df_model = df_model.merge(df_clients, on='account_id', how='left')
    df_model = df_model.merge(crm_agg, on='account_id', how='left')

    num_cols = df_model.select_dtypes(include=np.number).columns
    df_model[num_cols] = df_model[num_cols].fillna(0)
    cat_cols = df_model.select_dtypes(exclude=np.number).columns
    df_model[cat_cols] = df_model[cat_cols].fillna('UNKNOWN')

    # Personas
    df_model['receita_anualizada'] = df_model['monetary_total'] / 5.0
    def classificar_persona(row):
        if row['receita_anualizada'] >= 12200 and row['avg_days_between_purchases'] <= 120: return 'P1_Champions'
        elif row['receita_anualizada'] >= 12200 and row['avg_days_between_purchases'] > 120: return 'P2_Capex'
        elif row['receita_anualizada'] < 12200 and row['avg_days_between_purchases'] <= 120: return 'P3_Recorrentes'
        else: return 'P4_Cauda'
    df_model['persona_b2b'] = df_model.apply(classificar_persona, axis=1)
    
    df_processed = pd.get_dummies(df_model, columns=['segment', 'industry', 'canal_aquisicao', 'persona_b2b'], drop_first=True)
    return df_model, df_processed

@st.cache_resource
def carregar_modelo():
    # Ele tentará carregar o modelo de Personas original que treinamos
    try:
        return joblib.load('modelo_churn_personas.joblib')
    except Exception as e:
        st.error("Modelo .joblib não encontrado na pasta.")
        st.stop()

# --- Interface Web Streamlit ---
st.title("📊 Kovan LATAM - Monitor de Risco (Régua Dinâmica)")

with st.spinner("Carregando o modelo e aplicando regras dinâmicas..."):
    modelo = carregar_modelo()
    df_original, df_features = carregar_dados_dinamicos()

account_list = df_original['account_id'].sort_values().tolist()
selected_account = st.selectbox("Selecione o Cliente (Apenas clientes com >1 compra):", account_list)

if selected_account:
    conta_raw = df_original[df_original['account_id'] == selected_account].iloc[0]
    # Removemos o limite da base de predição porque o modelo não treinou com essa feature
    conta_features = df_features[df_features['account_id'] == selected_account].drop(columns=['account_id', 'limite_dinamico_dias', 'churn_label'], errors='ignore')
    
    colunas_modelo = modelo.feature_names_in_
    for col in colunas_modelo:
        if col not in conta_features.columns: conta_features[col] = 0
    conta_features = conta_features[colunas_modelo]
    
    prob_churn = modelo.predict_proba(conta_features)[0][1] * 100
    
    # Cálculos Dinâmicos
    recencia = conta_raw['recency_days']
    limite = conta_raw['limite_dinamico_dias']
    estouro = recencia - limite
    
    st.markdown("---")
    
    # --- MÓDULO: ALERTA DINÂMICO ---
    st.subheader("⏱️ Relógio de Compra (Régua Dinâmica)")
    col_a, col_b, col_c = st.columns(3)
    
    col_a.metric("Dias sem Comprar", f"{recencia:.0f} dias")
    col_b.metric("Limite de Tolerância (2x a Média)", f"{limite:.0f} dias")
    
    if estouro > 0:
        col_c.metric("Status do Limite", f"⚠️ Estourou {estouro:.0f} dias!", delta=f"-{estouro:.0f}", delta_color="inverse")
    else:
        folga = abs(estouro)
        col_c.metric("Status do Limite", f"✅ No Prazo", delta=f"+{folga:.0f} dias de folga", delta_color="normal")
        
    st.markdown("---")
    
    # --- MÓDULO: PERSONA & AÇÃO ---
    st.subheader("🎯 Perfil Estratégico (Matriz de Persona)")
    
    persona = conta_raw['persona_b2b']
    if persona == 'P1_Champions':
        st.success("**🟢 P1 · Champions Recorrentes** (Alto Valor / Alta Frequência)")
        st.write("**Ação:** Acionar KAM. Agendar QBR Trimestral.")
    elif persona == 'P2_Capex':
        st.warning("**🟡 P2 · Grandes Projetos Capex** (Alto Valor / Baixa Frequência)")
        st.write("**Ação:** Mapear próximo orçamento de TI. Upsell de Premier Support.")
    elif persona == 'P3_Recorrentes':
        st.info("**🟠 P3 · Pequenos Recorrentes** (Baixo Valor / Alta Frequência)")
        st.write("**Ação:** Automação CRM para Cross-sell e transformar em Multicesta.")
    else:
        st.error("**🔴 P4 · Transacionais de Cauda Spot** (Baixo Valor / Baixa Frequência)")
        st.write("**Ação:** Nurturing 100% digital. Redirecionar para E-commerce.")

    st.markdown("---")
    
    # --- MÓDULO: RISCO E MÉTRICAS ---
    st.subheader("⚠️ Avaliação de Risco (Machine Learning)")
    col1, col2, col3 = st.columns(3)
    
    if prob_churn > 70 or estouro > 0: status = "🔴 ALTO RISCO DE RUPTURA"
    elif prob_churn > 40: status = "🟡 MÉDIO RISCO"
    else: status = "🟢 RETENÇÃO SEGURA"
        
    col1.metric("Status Geral", status)
    col2.metric("Probabilidade do Algoritmo", f"{prob_churn:.1f}%")
    col3.metric("Frequência Média Histórica", f"{conta_raw['avg_days_between_purchases']:.0f} dias")
