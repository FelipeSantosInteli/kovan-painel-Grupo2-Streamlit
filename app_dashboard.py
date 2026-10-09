import streamlit as st
import pandas as pd
import numpy as np
from py_mini_racer import MiniRacer
import plotly.express as px

st.set_page_config(page_title="Kovan LATAM - Dashboard de Churn", layout="wide", page_icon="📊")

@st.cache_data
def carregar_e_processar_dados(uploaded_file):
    # 1. Leitura do Arquivo
    todas_as_abas = pd.read_excel(uploaded_file, sheet_name=None)
    
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

    # ==============================================================
    # PASSO A: DEFINIR A MATRIZ OFICIAL (BASE COMPLETA, SEM CORTE)
    # ==============================================================
    # Removemos os One-Offs para focar apenas nas 5.771 contas reais
    order_counts_full = df_raw.groupby('account_id')['invoice_no'].nunique()
    valid_accounts_full = order_counts_full[order_counts_full > 1].index
    df_raw_recorrentes = df_raw[df_raw['account_id'].isin(valid_accounts_full)].copy()

    rfm_full = df_raw_recorrentes.groupby('account_id').agg(monetary_total=('total_value_usd', 'sum')).reset_index()
    purchases_full = df_raw_recorrentes[['account_id', 'billing_dt']].drop_duplicates().sort_values(by=['account_id', 'billing_dt'])
    purchases_full['days_since_previous'] = purchases_full.groupby('account_id')['billing_dt'].diff().dt.days
    cadence_full = purchases_full.groupby('account_id')['days_since_previous'].mean().reset_index()
    cadence_full.columns = ['account_id', 'avg_days_between_purchases']
    
    df_personas = rfm_full.merge(cadence_full, on='account_id', how='left')
    df_personas['receita_anualizada'] = df_personas['monetary_total'] / 5.0
    
    def classificar_persona_oficial(row):
        if row['receita_anualizada'] >= 50000 and row['avg_days_between_purchases'] <= 30: return 'P1_Champions'
        elif row['receita_anualizada'] >= 50000 and row['avg_days_between_purchases'] > 120: return 'P2_Capex'
        elif row['receita_anualizada'] < 50000 and row['avg_days_between_purchases'] <= 60: return 'P3_Recorrentes'
        elif row['receita_anualizada'] < 50000 and row['avg_days_between_purchases'] <= 290: return 'P4_Cauda'
        
    df_personas['persona_b2b'] = df_personas.apply(classificar_persona_oficial, axis=1)
    df_personas_oficial = df_personas[['account_id', 'persona_b2b', 'receita_anualizada']]

    # ==============================================================
    # PASSO B: GERAR FEATURES PARA O MACHINE LEARNING (COM TRAVA)
    # ==============================================================
    CUTOFF_DATE = pd.to_datetime('2026-03-31') 
    df_features_period = df_raw[df_raw['billing_dt'] <= CUTOFF_DATE].copy()
    
    # Mantém apenas as contas recorrentes mapeadas acima
    df_features_period = df_features_period[df_features_period['account_id'].isin(valid_accounts_full)]

    rfm_div = df_features_period.groupby('account_id').agg(
        recency_date=('billing_dt', 'max'), frequency_orders=('invoice_no', 'nunique')
    ).reset_index()
    rfm_div['recency_days'] = (CUTOFF_DATE - rfm_div['recency_date']).dt.days

    purchase_dates = df_features_period[['account_id', 'billing_dt']].drop_duplicates().sort_values(by=['account_id', 'billing_dt'])
    purchase_dates['days_since_previous'] = purchase_dates.groupby('account_id')['billing_dt'].diff().dt.days
    cadence = purchase_dates.groupby('account_id')['days_since_previous'].agg(['mean', 'std']).reset_index()
    cadence.columns = ['account_id', 'avg_days_between_purchases', 'std_days_between_purchases']

    cadence['limite_dinamico_dias'] = (cadence['avg_days_between_purchases'] * 2).clip(lower=90)

    t_6m = CUTOFF_DATE - pd.DateOffset(months=6)
    rev_6m = df_features_period[df_features_period['billing_dt'] >= t_6m].groupby('account_id')['total_value_usd'].sum().reset_index().rename(columns={'total_value_usd': 'revenue_6m'})

    all_clients = pd.DataFrame({'account_id': valid_accounts_full})
    
    # Faz os Merges combinando a Persona Oficial com a Matemática Cortada
    df_model = all_clients.merge(df_personas_oficial, on='account_id', how='left')
    df_model = df_model.merge(rfm_div, on='account_id', how='left')
    df_model = df_model.merge(cadence, on='account_id', how='left')
    df_model = df_model.merge(rev_6m, on='account_id', how='left')
    df_model = df_model.merge(df_clients, on='account_id', how='left')
    df_model = df_model.merge(crm_agg, on='account_id', how='left')

    # Preencher números com 0
    num_cols = df_model.select_dtypes(include='number').columns
    df_model[num_cols] = df_model[num_cols].fillna(0)
    
    # Preencher categorias com UNKNOWN, mas EXCLUINDO as colunas de data (datetime)
    cat_cols = df_model.select_dtypes(exclude=['number', 'datetime']).columns
    df_model[cat_cols] = df_model[cat_cols].fillna('UNKNOWN')

    
    df_para_ml = df_model.drop(columns=['recency_date', 'limite_dinamico_dias', 'country', 'receita_anualizada'], errors='ignore')
    df_processed = pd.get_dummies(df_para_ml, columns=['segment', 'industry', 'canal_aquisicao', 'persona_b2b'], drop_first=True)
    
    # 2. Carregar Motor V8
    try:
        with open('modelo_churn_dinamico.js', 'r') as file:
            js_code = file.read()
        js_engine = MiniRacer()
        js_engine.eval(js_code)
    except Exception as e:
        st.error(f"Erro ao carregar o JS localmente: {e}")
        return None, None

    # 3. Calcular Probabilidades Globais
    probabilidades = []
    df_features_only = df_processed.drop(columns=['account_id'], errors='ignore').select_dtypes(exclude=['object', 'string', 'datetime64[ns]'])
    
    for _, row in df_features_only.iterrows():
        input_array = [float(x) for x in row.values]
        try:
            res = js_engine.call("score", input_array)
            probabilidades.append(res[1])
        except:
            probabilidades.append(0)
            
    df_model['prob_churn'] = probabilidades
    df_model['em_risco'] = df_model['prob_churn'] > 0.50
    
    df_model['data_estouro_estimada'] = df_model['recency_date'] + pd.to_timedelta(df_model['limite_dinamico_dias'], unit='D')
    df_model['mes_estouro'] = df_model['data_estouro_estimada'].dt.to_period('M').astype(str)

    return df_model, df_processed

# --- Interface Web ---
st.title("📊 Kovan LATAM - Dashboard B2B & Previsão de Churn")
st.markdown("*Plataforma de inteligência preditiva alimentada pelo modelo JavaScript (`modelo_churn_dinamico.js`)*")

st.sidebar.header("📁 Carga de Dados")
arquivo_upload = st.sidebar.file_uploader("Suba a base da Kovan (.xlsx)", type=["xlsx"])

if arquivo_upload is not None:
    with st.spinner("Processando base e executando modelo V8 em todos os clientes..."):
        df_original, df_features = carregar_e_processar_dados(arquivo_upload)
    
    if df_original is not None:
        st.markdown("### 💰 Impacto Financeiro Global (Net Revenue Retention)")
        
        receita_total_anual = df_original['receita_anualizada'].sum()
        df_risco = df_original[df_original['em_risco'] == True]
        receita_em_risco = df_risco['receita_anualizada'].sum()
        
        nrr_estimada = ((receita_total_anual - receita_em_risco) / receita_total_anual) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Receita Anual Total (ARR)", f"US$ {receita_total_anual/1e6:.2f}M")
        col2.metric("Receita Anual em Risco", f"US$ {receita_em_risco/1e6:.2f}M", delta=f"-{(receita_em_risco/receita_total_anual)*100:.1f}%", delta_color="inverse")
        col3.metric("Contas em Risco (>50%)", f"{len(df_risco)} contas")
        
        nrr_color = "🟢 Saudável" if nrr_estimada > 100 else "🔴 Risco de Contração" if nrr_estimada < 90 else "🟡 Alerta"
        col4.metric("NRR Projetado", f"{nrr_estimada:.1f}%", nrr_color)

        st.markdown("---")
        
        st.markdown("### 📈 Risco de Churn e Distribuição no Tempo")
        col_grafico1, col_grafico2 = st.columns(2)
        
        with col_grafico1:
            st.markdown("**Contas em Risco por Persona B2B**")
            risco_persona = df_risco.groupby('persona_b2b').size().reset_index(name='contas_em_risco')
            fig_bar = px.bar(risco_persona, x='persona_b2b', y='contas_em_risco', color='persona_b2b', text_auto=True,
                             labels={'persona_b2b': 'Persona', 'contas_em_risco': 'Qtd. Contas em Risco'},
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_grafico2:
            st.markdown("**Evolução Projetada de Rupturas (Por Mês de Estouro do SLA)**")
            evolucao = df_risco.groupby(['mes_estouro', 'persona_b2b']).size().reset_index(name='qtd_churns')
            evolucao = evolucao[evolucao['mes_estouro'] >= '2023-01']
            
            fig_line = px.line(evolucao, x='mes_estouro', y='qtd_churns', color='persona_b2b', markers=True,
                               labels={'mes_estouro': 'Mês Projetado do Estouro', 'qtd_churns': 'Volume de Contas Perdidas'})
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        st.markdown("### 🔎 Monitor de Risco por Conta (Deep-Dive)")
        account_list = df_original['account_id'].sort_values().tolist()
        selected_account = st.selectbox("Selecione uma conta específica para análise detalhada:", account_list)

        if selected_account:
            conta_raw = df_original[df_original['account_id'] == selected_account].iloc[0]
            
            prob_churn = conta_raw['prob_churn'] * 100
            recencia = conta_raw['recency_days']
            limite = conta_raw['limite_dinamico_dias']
            estouro = recencia - limite
            
            st.subheader("⏱️ Relógio de Compra")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Dias sem Comprar", f"{recencia:.0f} dias")
            col_b.metric("Limite da Régua", f"{limite:.0f} dias")
            if estouro > 0: col_c.metric("Status", f"⚠️ Estourou {estouro:.0f} dias", delta=f"-{estouro:.0f}", delta_color="inverse")
            else: col_c.metric("Status", f"✅ No Prazo", delta=f"+{abs(estouro):.0f} dias de folga", delta_color="normal")
                
            st.subheader("⚠️ Avaliação de Risco (Modelo JS)")
            col1, col2, col3 = st.columns(3)
            
            status = "🔴 ALTO RISCO" if prob_churn > 50 else "🟢 RETENÇÃO SEGURA"
            col1.metric("Status Geral", status)
            col2.metric("Probabilidade (.js)", f"{prob_churn:.1f}%")
            col3.metric("Persona B2B", conta_raw['persona_b2b'].replace('_', ' '))

else:
    st.info("👈 Por favor, faça o upload do dataset Excel no menu lateral para carregar o painel.")
