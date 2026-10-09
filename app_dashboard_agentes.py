import streamlit as st
import pandas as pd
import numpy as np
from py_mini_racer import MiniRacer
import plotly.express as px
import requests
import uuid

st.set_page_config(page_title="Kovan LATAM - Dashboard B2B", layout="wide", page_icon="📊")

CUTOFF_DATE = pd.to_datetime('2026-03-31') 


@st.cache_data
def carregar_e_processar_dados(uploaded_file):
    todas_as_abas = pd.read_excel(uploaded_file, sheet_name=None)
    
    df_raw, df_clients, df_crm = None, None, None
    for nome_aba, df_aba in todas_as_abas.items():
        col = df_aba.columns.tolist()
        if 'billing_dt' in col and 'total_value_usd' in col: df_raw = df_aba.copy()
        elif 'tempo_como_cliente' in col and 'canal_aquisicao' in col: df_clients = df_aba.copy()
        elif 'contatos_realizados' in col and 'oportunidades_abertas' in col: df_crm = df_aba.copy()

    df_raw['billing_dt'] = pd.to_datetime(df_raw['billing_dt'])
    df_clients = df_clients[['account_id', 'segment', 'industry', 'canal_aquisicao', 'tempo_como_cliente']].drop_duplicates()
    
    crm_agg = df_crm.groupby('account_id').agg(
        contatos_realizados=('contatos_realizados', 'sum'),
        oportunidades_abertas=('oportunidades_abertas', 'sum'),
        oportunidades_ganhas=('oportunidades_ganhas', 'sum'),
        dias_sem_contato=('dias_sem_contato', 'max')
    ).reset_index()
    
    crm_agg['win_rate'] = crm_agg['oportunidades_ganhas'] / crm_agg['oportunidades_abertas']
    crm_agg['win_rate'] = crm_agg['win_rate'].fillna(0).replace([np.inf, -np.inf], 0)

    # =========================================================================
    # 1. VISÃO DE NEGÓCIO (5 ANOS) - CRAVANDO QUANTIDADES E RECEITA TOTAL
    # =========================================================================
    rfm_full = df_raw.groupby('account_id').agg(
        monetary_total=('total_value_usd', 'sum'),
        invoice_count=('invoice_no', 'nunique')
    ).reset_index()
    
    # Faturamento Anualizado Real
    rfm_full['receita_anualizada'] = rfm_full['monetary_total'] / 5.0
    
    purchases_full = df_raw[['account_id', 'billing_dt']].drop_duplicates().sort_values(by=['account_id', 'billing_dt'])
    purchases_full['days_since_previous'] = purchases_full.groupby('account_id')['billing_dt'].diff().dt.days
    cadence_full = purchases_full.groupby('account_id')['days_since_previous'].mean().reset_index()
    cadence_full.columns = ['account_id', 'avg_days_between_purchases']
    
    df_personas = rfm_full.merge(cadence_full, on='account_id', how='left')
    
    # Classificação Oficial da Matriz B2B da Kovan
    def classificar_persona(row):
        if row['invoice_count'] <= 1: 
            return 'One-Offs'
        elif row['receita_anualizada'] >= 50000 and row['avg_days_between_purchases'] <= 30: 
            return 'P1_Champions'
        elif row['receita_anualizada'] >= 50000 and row['avg_days_between_purchases'] > 120: 
            return 'P2_Capex'
        elif row['receita_anualizada'] < 50000 and row['avg_days_between_purchases'] <= 90: 
            return 'P3_Recorrentes'
        else: 
            return 'P4_Cauda'
            
    df_personas['persona_b2b'] = df_personas.apply(classificar_persona, axis=1)
    
    # SLA Preventivo Oficial (Política Fixa)
    def definir_sla(persona):
        slas = {'P1_Champions': 20, 'P2_Capex': 150, 'P3_Recorrentes': 40, 'P4_Cauda': 290}
        return slas.get(persona, 9999) # Ignora SLA para One-Offs
        
    df_personas['limite_sla_dias'] = df_personas['persona_b2b'].apply(definir_sla)

    # =========================================================================
    # 2. MOTOR DE MACHINE LEARNING (PREPARAÇÃO COM TRAVA TEMPORAL)
    # =========================================================================
    df_features_period = df_raw[df_raw['billing_dt'] <= CUTOFF_DATE].copy()

    rfm_ml = df_features_period.groupby('account_id').agg(
        recency_date=('billing_dt', 'max'),
        frequency_orders=('invoice_no', 'nunique')
    ).reset_index()
    rfm_ml['recency_days'] = (CUTOFF_DATE - rfm_ml['recency_date']).dt.days

    t_6m = CUTOFF_DATE - pd.DateOffset(months=6)
    rev_6m = df_features_period[df_features_period['billing_dt'] >= t_6m].groupby('account_id')['total_value_usd'].sum().reset_index().rename(columns={'total_value_usd': 'revenue_6m'})

    df_model = df_personas.merge(rfm_ml, on='account_id', how='left')
    df_model = df_model.merge(rev_6m, on='account_id', how='left')
    df_model = df_model.merge(df_clients, on='account_id', how='left')
    df_model = df_model.merge(crm_agg, on='account_id', how='left')

    df_model['std_days_between_purchases'] = 0
    df_model['avg_days_between_purchases'] = df_model['avg_days_between_purchases'].fillna(9999)

    num_cols = df_model.select_dtypes(include='number').columns
    df_model[num_cols] = df_model[num_cols].fillna(0)
    
    return df_model, df_raw

@st.cache_resource
def carregar_motor_javascript():
    try:
        with open('modelo_churn_dinamico.js', 'r') as file:
            js_code = file.read()
        ctx = MiniRacer()
        ctx.eval(js_code)
        return ctx
    except Exception as e:
        return None

# --- INTERFACE WEB STREAMLIT ---
st.title("📊 Kovan LATAM - Dashboard B2B & Previsão de Churn")
st.markdown("*Inteligência Preditiva aplicada às regras de negócio B2B oficiais da Kovan*")

st.sidebar.header("📁 Carga de Dados")
arquivo_upload = st.sidebar.file_uploader("Suba a base da Kovan (.xlsx)", type=["xlsx"])

if arquivo_upload is not None:
    with st.spinner("Processando base de 5 anos e injetando SLA oficial..."):
        df_original, df_raw = carregar_e_processar_dados(arquivo_upload)
        js_engine = carregar_motor_javascript()
        
        if df_original is not None and js_engine is not None:
            probabilidades = []
            
            # Avaliação individual no Motor JS
            for _, row in df_original.iterrows():
                input_array = [
                    float(row['frequency_orders']), float(row['monetary_total']), float(row['recency_days']),
                    float(row['avg_days_between_purchases']), float(row['std_days_between_purchases']), float(row['revenue_6m']),
                    float(row['tempo_como_cliente']), float(row['contatos_realizados']), float(row['oportunidades_abertas']),
                    float(row['oportunidades_ganhas']), float(row['dias_sem_contato']), float(row['win_rate']),
                    float(row['receita_anualizada']),
                    1.0 if row.get('segment') == 'LARGE ENTERPRISE' else 0.0, 1.0 if row.get('segment') == 'MID MARKET' else 0.0,
                    1.0 if row.get('segment') == 'PUBLIC SECTOR' else 0.0, 1.0 if row.get('segment') == 'SMALL MARKET' else 0.0,
                    1.0 if row.get('segment') == 'STRATEGIC ACCOUNT' else 0.0,
                    1.0 if row.get('industry') == 'COMMUNICATIONS, MEDIA AND SERVICES' else 0.0, 1.0 if row.get('industry') == 'EDUCATION' else 0.0,
                    1.0 if row.get('industry') == 'GOVERNMENT' else 0.0, 1.0 if row.get('industry') == 'HEALTHCARE AND LIFE SCIENCES' else 0.0,
                    1.0 if row.get('industry') == 'INSURANCE' else 0.0, 1.0 if row.get('industry') == 'MANUFACTURING AND NATURAL RESOURCES' else 0.0,
                    1.0 if row.get('industry') == 'OIL AND GAS' else 0.0, 1.0 if row.get('industry') == 'POWER AND UTILITIES' else 0.0,
                    1.0 if row.get('industry') == 'RETAIL' else 0.0, 1.0 if row.get('industry') == 'TRANSPORTATION' else 0.0,
                    1.0 if row.get('industry') == 'UNCLASSIFIED' else 0.0, 1.0 if row.get('industry') == 'UNKNOWN' else 0.0,
                    1.0 if row.get('industry') == 'WHOLESALE TRADE' else 0.0,
                    1.0 if row.get('persona_b2b') == 'P2_Capex' else 0.0, 1.0 if row.get('persona_b2b') == 'P3_Recorrentes' else 0.0,
                    1.0 if row.get('persona_b2b') == 'P4_Cauda' else 0.0
                ]
                try:
                    res = js_engine.call("score", input_array)
                    probabilidades.append(res[1])
                except:
                    probabilidades.append(0)
                    
            df_original['prob_churn'] = probabilidades
            
            # --- AJUSTE DE LIMITE: RISCO APENAS >= 75% e IGNORAR ONE-OFFS ---
            df_recorrentes = df_original[df_original['persona_b2b'] != 'One-Offs'].copy()

            # 1. Calculamos quantos dias o cliente está inativo com base na CUTOFF_DATE
            df_recorrentes['dias_inativos'] = (CUTOFF_DATE - df_recorrentes['recency_date']).dt.days

            # 2. O cliente SÓ está em risco se ele ultrapassou o seu limite oficial (Estouro de SLA)
            df_recorrentes['em_risco'] = df_recorrentes['dias_inativos'] > df_recorrentes['limite_sla_dias']

            # df_recorrentes['em_risco'] = df_recorrentes['prob_churn'] >= 0.75
            
            df_recorrentes['data_estouro_estimada'] = df_recorrentes['recency_date'] + pd.to_timedelta(df_recorrentes['limite_sla_dias'], unit='D')
            df_recorrentes['mes_estouro'] = df_recorrentes['data_estouro_estimada'].dt.to_period('M').astype(str)

            # --- MÓDULO 1: RESUMO FINANCEIRO GLOBAL ---
            st.markdown("### 💰 Impacto Financeiro Global (Net Revenue Retention)")
            
            # --- NOVO CÁLCULO DE NRR B2B (Ano 4 vs Ano 5) ---
            cond_ano4 = (df_raw['billing_dt'] >= '2024-04-01') & (df_raw['billing_dt'] <= '2025-03-31')
            cond_ano5 = (df_raw['billing_dt'] >= '2025-04-01') & (df_raw['billing_dt'] <= '2026-03-31')
            
            faturamento_ano4 = df_raw[cond_ano4].groupby('account_id')['total_value_usd'].sum()
            faturamento_ano5 = df_raw[cond_ano5].groupby('account_id')['total_value_usd'].sum()
            
            # Isola a base exata de clientes ativos no Ano 4
            contas_ano4 = faturamento_ano4[faturamento_ano4 > 0].index
            
            receita_base_ano4 = faturamento_ano4[contas_ano4].sum()
            receita_retida_ano5 = faturamento_ano5.reindex(contas_ano4).fillna(0).sum()
            
            nrr_estimada = (receita_retida_ano5 / receita_base_ano4) * 100 if receita_base_ano4 > 0 else 0
            
            # --- NOVO CÁLCULO DE ARR (Receita Real dos Últimos 12 Meses) ---
            receita_total_anual = df_raw[cond_ano5]['total_value_usd'].sum()
            
            # Calculando a Receita em Risco (mantendo a lógica preditiva)
            df_risco = df_recorrentes[df_recorrentes['em_risco'] == True]
            receita_em_risco = df_risco['receita_anualizada'].sum()

            df_contas_risco = df_recorrentes[(df_recorrentes['em_risco'] == True) & (df_recorrentes['limite_sla_dias'] <= 150)]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Receita Anual Total (ARR)", f"US$ {receita_total_anual/1e6:.2f}M")
            col2.metric("Receita Anual em Risco", f"US$ {receita_em_risco/1e6:.2f}M", delta=f"-{(receita_em_risco/receita_total_anual)*100:.2f}%", delta_color="inverse")
            col3.metric("Contas em Risco (>=75%)", f"{len(df_contas_risco)} contas")
            
            nrr_color = "🟢 Saudável" if nrr_estimada > 100 else "🔴 Risco de Contração" if nrr_estimada < 90 else "🟡 Alerta"
            col4.metric("NRR Projetado", f"{nrr_estimada:.2f}%", nrr_color)

            st.markdown("---")
            
            # --- MÓDULO 2: VISÃO POR PERSONA E EVOLUÇÃO (BARRAS / 1 ANO) ---
            st.markdown("### 📈 Risco de Churn e Distribuição (Projeção 2025)")
            col_grafico1, col_grafico2 = st.columns(2)
            
            with col_grafico1:
                st.markdown("**Contas em Risco por Persona B2B**")
                risco_persona = df_risco.groupby('persona_b2b').size().reset_index(name='contas_em_risco')
                fig_bar_persona = px.bar(risco_persona, x='persona_b2b', y='contas_em_risco', color='persona_b2b', text_auto=True,
                                 labels={'persona_b2b': 'Persona', 'contas_em_risco': 'Contas em Risco'})
                fig_bar_persona.update_layout(showlegend=False)
                st.plotly_chart(fig_bar_persona, use_container_width=True)
                
            with col_grafico2:
                st.markdown("**Evolução de Rupturas de SLA (Visão 2025)**")
                # RECORTE ESTRITO: Apenas meses do ano de 2025
                df_grafico = df_risco[(df_risco['mes_estouro'] >= '2025-01') & (df_risco['mes_estouro'] <= '2025-12')]
                
                evolucao = df_grafico.groupby(['mes_estouro', 'persona_b2b']).size().reset_index(name='qtd_churns')
                
                # Gráfico de barras agrupadas em vez de linha
                fig_bar_tempo = px.bar(evolucao, x='mes_estouro', y='qtd_churns', color='persona_b2b', barmode='group',
                                   labels={'mes_estouro': 'Mês Projetado do Estouro', 'qtd_churns': 'Volume Perdido', 'persona_b2b': 'Persona'})
                st.plotly_chart(fig_bar_tempo, use_container_width=True)

            st.markdown("---")
            
            # --- MÓDULO 3: MONITOR INDIVIDUAL ---
            st.markdown("### 🔎 Monitor de Risco por Conta (Deep-Dive)")
            account_list = df_recorrentes['account_id'].sort_values().tolist()
            selected_account = st.selectbox("Selecione uma conta recorrente para análise detalhada:", account_list)

            if selected_account:
                conta_raw = df_recorrentes[df_recorrentes['account_id'] == selected_account].iloc[0]
                prob_churn = conta_raw['prob_churn'] * 100
                recencia = conta_raw['recency_days']
                limite = conta_raw['limite_sla_dias'] # Agora usa rigorosamente a regra de negócio (30, 150, 90, 180)
                estouro = recencia - limite
                
                st.subheader("⏱️ Relógio de Compra B2B")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Dias sem Comprar", f"{recencia:.0f} dias")
                col_b.metric("SLA de Retenção", f"{limite:.0f} dias")
                if estouro > 0: col_c.metric("Status Operacional", f"⚠️ Estourou {estouro:.0f} dias", delta=f"-{estouro:.0f}", delta_color="inverse")
                else: col_c.metric("Status Operacional", f"✅ No Prazo", delta=f"+{abs(estouro):.0f} dias de folga", delta_color="normal")
                    
                st.subheader("⚠️ Avaliação de Risco Preditivo (.js)")
                col1, col2, col3 = st.columns(3)
                status = "🔴 ALTO RISCO" if prob_churn >= 75 else "🟢 RETENÇÃO SEGURA"
                col1.metric("Alerta CRM", status)
                col2.metric("Probabilidade (.js)", f"{prob_churn:.1f}%")
                col3.metric("Persona B2B", conta_raw['persona_b2b'].replace('_', ' '))

                # =========================================================================
                # INTEGRAÇÃO COM N8N (CARD 3 E CARD 4 DO HTML ORIGINAL)
                # =========================================================================
                st.markdown("---")
                st.markdown("### 🔌 Conexão e Chat com a Equipe n8n")

                # Controle de estado para as mensagens do chat e ID da sessão
                if "mensagens" not in st.session_state:
                    st.session_state.mensagens = [{"role": "assistant", "nome": "Sistema", "content": "Carregue a planilha e configure o endpoint para começar."}]
                if "sessao_id" not in st.session_state:
                    st.session_state.sessao_id = f"painel-{uuid.uuid4().hex[:8]}"

                col_config, col_chat = st.columns([1, 2])

                with col_config:
                    st.markdown("#### Configurações (Card 3)")
                    endpoint_n8n = st.text_input("Endpoint do Webhook (n8n)", placeholder="https://...")
                    modelo_llm = st.selectbox(
                        "Modelo de Linguagem", 
                        ["nvidia/nemotron-3-super-120b-a12b:free", "qwen/qwen3.8-27b:free", "outro"]
                    )
                    
                    with st.expander("Planos de Ação (Para a Agente Vera)"):
                        plano_comercial = st.text_area("Comercial", "1. Conta com valor esperado acima de USD 500 mil: ligação do Account Manager...\n2. Conta com receita em queda: proposta de renovação...")
                        plano_atendimento = st.text_area("Atendimento", "1. Conta com 4 meses sem compra: contato do time de atendimento...\n2. Registrar no CRM...")
                        plano_posvendas = st.text_area("Pós-vendas", "1. Revisão técnica do parque...\n2. Oferta de extensão de garantia...")

                with col_chat:
                    st.markdown("#### Conversa com a Equipe (Card 4)")
                    
                    # Container para histórico do chat
                    chat_container = st.container(height=500)
                    with chat_container:
                        for msg in st.session_state.mensagens:
                            with st.chat_message(msg["role"]):
                                if "nome" in msg:
                                    st.markdown(f"**{msg['nome']}**\n\n{msg['content']}")
                                else:
                                    st.markdown(msg["content"])
                    
                    # Campo de input de chat na parte inferior
                    if prompt := st.chat_input("Pergunte sobre a fila, uma conta ou o plano de uma área..."):
                        if not endpoint_n8n:
                            st.warning("⚠️ Por favor, configure o Endpoint do Webhook primeiro.")
                        else:
                            # 1. Adiciona e mostra a mensagem do usuário
                            st.session_state.mensagens.append({"role": "user", "nome": "Você", "content": prompt})
                            with chat_container:
                                with st.chat_message("user"):
                                    st.markdown(f"**Você**\n\n{prompt}")
                            
                            # 2. Prepara os dados no formato que os prompts do n8n esperam
                            df_para_n8n = df_risco.copy()
                            contas_formatadas = []
                            for i, row in enumerate(df_para_n8n.head(150).itertuples()):
                                contas_formatadas.append({
                                    "posicao_na_fila": i + 1,
                                    "account_id": row.account_id,
                                    "segmento": getattr(row, 'persona_b2b', 'N/A'),
                                    "escore_de_perda": round(getattr(row, 'prob_churn', 0), 3),
                                    "receita_12m_usd": round(getattr(row, 'receita_anualizada', 0)),
                                    "meses_desde_ultima_compra": round(getattr(row, 'dias_inativos', 0) / 30)
                                })
                        
                        conta_foco_payload = None
                        
                        # Se uma conta foi selecionada no painel, buscamos na aba 'raw_data' (df_raw)
                        if 'selected_account' in locals() and selected_account:
                            metricas_conta = df_recorrentes[df_recorrentes['account_id'] == selected_account]
                            
                            # Busca na aba raw_data todas as compras desse cliente específico
                            historico_cru = df_raw[df_raw['account_id'] == selected_account].copy()
                            historico_cru = historico_cru.sort_values(by='billing_dt', ascending=False)
                            
                            # Formata a data para texto (YYYY-MM-DD) para não quebrar o JSON do n8n
                            historico_cru['billing_dt'] = historico_cru['billing_dt'].dt.strftime('%Y-%m-%d')
                            
                            if not metricas_conta.empty:
                                conta_foco_payload = {
                                    "account_id": selected_account,
                                    "persona": metricas_conta['persona_b2b'].values[0],
                                    "risco_calculado": float(metricas_conta['prob_churn'].values[0]) if 'prob_churn' in metricas_conta else "N/A",
                                    "dias_inativos_atuais": int(metricas_conta['dias_inativos'].values[0]),
                                    "limite_sla_permitido": int(metricas_conta['limite_sla_dias'].values[0]),
                                    # A IA vai ler exatamente as colunas da sua aba raw_data
                                    "historico_transacoes_raw": historico_cru[['invoice_no', 'billing_dt', 'total_value_usd']].astype(str).to_dict('records')
                                }
                            
                            payload = {
                                "sessionId": st.session_state.sessao_id,
                                "pergunta": prompt,
                                "modelo_llm": modelo_llm,
                                "conta_em_foco": conta_foco_payload, 
                                "modelo": {
                                    "contas_elegiveis": len(df_recorrentes), 
                                    "capacidade_do_ciclo": len(contas_formatadas)
                                },
                                "contas_gerais_fila": contas_formatadas,
                                "planos": {
                                    "comercial": plano_comercial,
                                    "atendimento": plano_atendimento,
                                    "pós-vendas": plano_posvendas
                                }
                            }
                            
                            # 3. Chama o webhook do n8n
                            with chat_container:
                                with st.spinner("A equipe (Atlas, Vera e Ciro) está analisando..."):
                                    try:
                                        resposta = requests.post(endpoint_n8n, json=payload, timeout=120)
                                        resposta.raise_for_status()
                                        dados = resposta.json()
                                        
                                        # Processa e adiciona as respostas de cada agente
                                        agentes = dados.get('agentes', [])
                                        if not agentes: # Fallback caso a API retorne num formato diferente
                                            agentes = [{"nome": "Sistema", "papel": "", "texto": str(dados)}]
                                            
                                        for agente in agentes:
                                            texto_formatado = f"*{agente.get('papel', '')}*\n\n{agente.get('texto', '')}"
                                            st.session_state.mensagens.append({
                                                "role": "assistant", 
                                                "nome": agente.get('nome', 'Agente'), 
                                                "content": texto_formatado
                                            })
                                            with st.chat_message("assistant"):
                                                st.markdown(f"**{agente.get('nome', 'Agente')}**\n\n{texto_formatado}")
                                                
                                    except Exception as e:
                                        st.error(f"Erro na conexão com o n8n: {e}")


else:
    st.info("👈 Por favor, faça o upload do dataset Excel no menu lateral para carregar o painel.")
