"""
Monta o payload enviado ao webhook n8n (Atlas -> Vera -> Gilberto -> Ciro).

Principios:
- Tudo que o agente precisa ja chega CALCULADO (receita 12m, queda vs. pico, valor esperado...).
  Os agentes nao devem somar nem estimar nada a partir de linhas brutas.
- Todo dado respeita a data de corte (CUTOFF_DATE). Compras posteriores ao corte nao entram.
- Nenhum NaN/inf no JSON (requests rejeita NaN).
"""
import numpy as np
import pandas as pd

CUTOFF_DATE = pd.to_datetime("2026-03-31")
INICIO_12M = (CUTOFF_DATE + pd.Timedelta(days=1)) - pd.DateOffset(years=1)  # 2025-04-01
LIMITE_INATIVO_DIAS = 365      # sem comprar ha mais de 1 ano = inativo
MAX_TRANSACOES_HISTORICO = 40  # linhas brutas mais recentes enviadas ao Gilberto
MAX_FAMILIAS = 8


# ---------------------------------------------------------------- utilitarios
def _num(x, nd=2):
    """float seguro para JSON (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, nd)


def _data(x):
    return None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def definir_status(dias_inativos, limite_sla):
    """ativo = dentro do SLA | em_risco = estourou o SLA | inativo = > 365 dias sem comprar."""
    if dias_inativos > LIMITE_INATIVO_DIAS:
        return "inativo"
    if dias_inativos > limite_sla:
        return "em_risco"
    return "ativo"


# ------------------------------------------------------- metricas de receita
def calcular_metricas_receita(df_raw):
    """
    Por conta, usando janelas de 12 meses fechadas em marco (abr-mar), ate o corte:
      receita_12m_usd, receita_12m_anteriores_usd, pico_anual_usd,
      fracao_do_pico, razao_vs_12m_anteriores, e a receita por janela.
    """
    d = df_raw[df_raw["billing_dt"] <= CUTOFF_DATE].copy()
    d["fy"] = d["billing_dt"].dt.year + (d["billing_dt"].dt.month >= 4).astype(int)
    fy_atual = CUTOFF_DATE.year  # janela abr/2025 - mar/2026 => 2026
    anos = list(range(fy_atual - 4, fy_atual + 1))

    anual = (
        d.pivot_table(index="account_id", columns="fy", values="total_value_usd",
                      aggfunc="sum", fill_value=0.0)
        .reindex(columns=anos, fill_value=0.0)
    )
    m = pd.DataFrame(index=anual.index)
    m["receita_12m_usd"] = anual[fy_atual]
    m["receita_12m_anteriores_usd"] = anual[fy_atual - 1]
    m["pico_anual_usd"] = anual.max(axis=1)
    m["fracao_do_pico"] = np.where(m["pico_anual_usd"] > 0,
                                   m["receita_12m_usd"] / m["pico_anual_usd"], np.nan)
    m["razao_vs_12m_anteriores"] = np.where(m["receita_12m_anteriores_usd"] > 0,
                                            m["receita_12m_usd"] / m["receita_12m_anteriores_usd"], np.nan)
    rotulos = {a: f"{a - 1}-04 a {a}-03" for a in anos}
    m["_janelas"] = [
        {rotulos[a]: _num(v) for a, v in zip(anos, linha)} for linha in anual.values
    ]
    return m


# ---------------------------------------------------------- resumo de produtos
def _resumo_grupo(h, h12, coluna, limite=None):
    if h.empty:
        return []
    g = h.groupby(coluna).agg(
        receita_total_usd=("total_value_usd", "sum"),
        pedidos=("invoice_no", "nunique"),
        ultima_compra=("billing_dt", "max"),
    )
    g["receita_12m_usd"] = h12.groupby(coluna)["total_value_usd"].sum().reindex(g.index).fillna(0.0)
    total = g["receita_total_usd"].sum()
    g["pct_da_receita_total"] = g["receita_total_usd"] / total if total > 0 else np.nan
    g = g.sort_values("receita_total_usd", ascending=False)
    if limite:
        g = g.head(limite)
    return [
        {
            coluna: idx,
            "receita_total_usd": _num(r.receita_total_usd),
            "receita_12m_usd": _num(r.receita_12m_usd),
            "pct_da_receita_total": _num(r.pct_da_receita_total, 3),
            "pedidos": int(r.pedidos),
            "ultima_compra": _data(r.ultima_compra),
        }
        for idx, r in g.iterrows()
    ]


# ---------------------------------------------------------------- conta em foco
def montar_conta_foco(account_id, df_recorrentes, df_raw, metricas, posicao_fila):
    linha = df_recorrentes[df_recorrentes["account_id"] == account_id]
    if linha.empty:
        return None
    c = linha.iloc[0]

    h = df_raw[(df_raw["account_id"] == account_id) & (df_raw["billing_dt"] <= CUTOFF_DATE)]
    h12 = h[h["billing_dt"] >= INICIO_12M]
    met = metricas.loc[account_id] if account_id in metricas.index else None

    dias_inativos = int(c["dias_inativos"])
    limite = int(c["limite_sla_dias"])
    receita_12m = _num(met["receita_12m_usd"]) if met is not None else 0.0
    prob = _num(c["prob_churn"], 3)
    valor_esperado = _num(prob * receita_12m) if (prob is not None and receita_12m is not None) else None

    ultimas = h.sort_values("billing_dt", ascending=False).head(MAX_TRANSACOES_HISTORICO)
    historico = [
        {
            "invoice_no": r.invoice_no,
            "billing_dt": _data(r.billing_dt),
            "brand_lenovo": r.brand_lenovo,
            "family_lenovo": r.family_lenovo,
            "series_lenovo": r.series_lenovo,
            "quantity": int(r.quantity),
            "total_value_usd": _num(r.total_value_usd),
        }
        for r in ultimas.itertuples()
    ]

    return {
        "account_id": account_id,
        "persona": c["persona_b2b"],
        "segmento": c.get("segment"),
        "industria": c.get("industry"),
        "pais": h["country"].iloc[0] if not h.empty else None,
        "tempo_como_cliente_meses": _num(c.get("tempo_como_cliente"), 1),
        "status_cliente": definir_status(dias_inativos, limite),
        "posicao_na_fila": posicao_fila,  # None se a conta nao esta na fila de risco
        "escore_de_perda": prob,
        "dias_inativos_atuais": dias_inativos,
        "meses_desde_ultima_compra": _num(dias_inativos / 30, 1),
        "data_ultima_compra": _data(c["recency_date"]),
        "limite_sla_dias": limite,
        "dias_ate_estouro_sla": limite - dias_inativos,  # negativo = SLA ja estourado
        "receita_12m_usd": receita_12m,
        "receita_12m_anteriores_usd": _num(met["receita_12m_anteriores_usd"]) if met is not None else None,
        "valor_esperado_usd": valor_esperado,
        "receita_12m_como_fracao_do_pico_anual": _num(met["fracao_do_pico"], 3) if met is not None else None,
        "receita_12m_dividida_pela_dos_12m_anteriores": _num(met["razao_vs_12m_anteriores"], 3) if met is not None else None,
        "receita_por_janela_12m_usd": met["_janelas"] if met is not None else None,
        "receita_total_historico_usd": _num(h["total_value_usd"].sum()),
        "total_pedidos_historico": int(h["invoice_no"].nunique()),
        "resumo_por_marca": _resumo_grupo(h, h12, "brand_lenovo"),
        "resumo_por_familia": _resumo_grupo(h, h12, "family_lenovo", MAX_FAMILIAS),
        "total_linhas_historico": int(len(h)),
        "historico_transacoes_raw": historico,  # so as N mais recentes; NAO somar
    }


# --------------------------------------------------------------------- payload
def construir_payload(*, session_id, pergunta, modelo_llm, selected_account,
                      df_recorrentes, df_risco, df_contas_risco, df_raw,
                      receita_total_anual, receita_em_risco_anualizada, nrr,
                      planos, tamanho_fila=30):
    metricas = calcular_metricas_receita(df_raw)

    # ---- fila: contas em risco ordenadas por valor esperado (criterio do painel)
    fila = df_risco.copy()
    fila["receita_12m_usd"] = fila["account_id"].map(metricas["receita_12m_usd"]).fillna(0.0)
    fila["valor_esperado_usd"] = fila["prob_churn"] * fila["receita_12m_usd"]
    fila = fila.sort_values("valor_esperado_usd", ascending=False).reset_index(drop=True)

    receita_12m_em_risco = float(fila["receita_12m_usd"].sum())
    posicoes = {acc: i + 1 for i, acc in enumerate(fila["account_id"])}

    contas = []
    for i, r in enumerate(fila.head(tamanho_fila).itertuples()):
        m = metricas.loc[r.account_id]
        dias = int(r.dias_inativos)
        contas.append({
            "posicao_na_fila": i + 1,
            "account_id": r.account_id,
            "persona": r.persona_b2b,
            "segmento": getattr(r, "segment", None),
            "status_cliente": definir_status(dias, int(r.limite_sla_dias)),
            "escore_de_perda": _num(r.prob_churn, 3),
            "receita_12m_usd": _num(r.receita_12m_usd),
            "valor_esperado_usd": _num(r.valor_esperado_usd),
            "receita_12m_como_fracao_do_pico_anual": _num(m["fracao_do_pico"], 3),
            "receita_12m_dividida_pela_dos_12m_anteriores": _num(m["razao_vs_12m_anteriores"], 3),
            "meses_desde_ultima_compra": _num(dias / 30, 1),
            "limite_sla_dias": int(r.limite_sla_dias),
        })

    modelo = {
        "data_de_corte": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "janela_12m": f"{INICIO_12M:%Y-%m-%d} a {CUTOFF_DATE:%Y-%m-%d}",
        "criterio_conta_em_risco": "dias sem comprar maior que o limite_sla_dias da persona (nao e o escore)",
        "criterio_status_cliente": f"ativo: dentro do SLA; em_risco: estourou o SLA; inativo: mais de {LIMITE_INATIVO_DIAS} dias sem comprar",
        "contas_elegiveis": int(len(df_recorrentes)),
        "contas_em_risco_total": int(len(df_risco)),
        "contas_em_risco_sla_ate_150d": int(len(df_contas_risco)),
        "contas_na_fila_enviada": len(contas),
        "receita_total_12m_usd": _num(receita_total_anual),
        "receita_12m_das_contas_em_risco_usd": _num(receita_12m_em_risco),
        "receita_em_risco_anualizada_5a_usd": _num(receita_em_risco_anualizada),
        "nrr_b2b_percentual": _num(nrr),
    }

    conta_foco = None
    if selected_account:
        conta_foco = montar_conta_foco(selected_account, df_recorrentes, df_raw, metricas,
                                       posicoes.get(selected_account))

    marcas = sorted(
        b for b in df_raw["brand_lenovo"].dropna().unique() if b != "WITHOUT INFO PRODUCT"
    )

    return {
        "sessionId": session_id,
        "pergunta": pergunta,
        "modelo_llm": modelo_llm,
        "conta_em_foco": conta_foco,
        "modelo": modelo,
        "contas": contas,
        "catalogo_marcas": marcas,
        "planos": planos,
    }
