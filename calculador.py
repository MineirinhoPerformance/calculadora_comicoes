import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import altair as alt

# -------------------------------------------------------
# Configurar a página (primeiro comando do Streamlit)
# -------------------------------------------------------
st.set_page_config(page_title="Calculadora de Comissões por KG", layout="wide")

# -------------------------------------------------------
# Funções auxiliares
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_db_engine():
    # As credenciais do banco (host, database, user, password, port)
    # devem ser definidas no arquivo de segredos (secrets.toml) ou no painel de Secrets do Streamlit Cloud.
    cfg = st.secrets["mysql"]
    connection_string = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(connection_string)

@st.cache_data(show_spinner=False)
def load_faturamento(df_fatur):
    df = df_fatur.copy()
    df.columns = df.columns.str.strip().str.lower()
    df['emissao'] = pd.to_datetime(df['emissao'], dayfirst=True, errors='coerce')
    df['unid_faturado'] = pd.to_numeric(df['unid_faturado'], errors='coerce').fillna(0)
    df['total df'] = pd.to_numeric(df['total df'], errors='coerce').fillna(0.0)
    df['total kg'] = pd.to_numeric(df['total kg'], errors='coerce').fillna(0.0)
    df['ano'] = df['emissao'].dt.year
    df['mes'] = df['emissao'].dt.month
    df['nome_distribuidor'] = df['nome_distribuidor'].astype(str)
    df['codigo_produto'] = df['codigo_produto'].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_meta_filtered(dist, ano, mes):
    """
    Busca metas mensais (KG e R$) por distribuidor (sem relação a produto).
    Ajusta nomes para uppercase+trim para casar com Excel.
    """
    engine = get_db_engine()
    dist_upper = [d.strip().upper() for d in dist]
    dist_clause = ",".join(f"'{d}'" for d in dist_upper)

    query_kg = f"""
        SELECT 
          UPPER(TRIM(nome_dist)) AS nome_distribuidor,
          SUM(valor_meta) AS meta_kg_dist
        FROM adms_metas
        WHERE UPPER(TRIM(nome_dist)) IN ({dist_clause})
          AND YEAR(`data`) = {ano}
          AND MONTH(`data`) = {mes}
        GROUP BY UPPER(TRIM(nome_dist))
    """
    query_rs = f"""
        SELECT 
          UPPER(TRIM(nome_dist)) AS nome_distribuidor,
          SUM(valor_meta) AS meta_r_dist
        FROM adms_metas_R$
        WHERE UPPER(TRIM(nome_dist)) IN ({dist_clause})
          AND YEAR(`data`) = {ano}
          AND MONTH(`data`) = {mes}
        GROUP BY UPPER(TRIM(nome_dist))
    """

    df_meta_kg = pd.read_sql(query_kg, engine)
    df_meta_rs = pd.read_sql(query_rs, engine)

    # Preencher distribuidores ausentes com zero
    for d in dist_upper:
        if d not in df_meta_kg['nome_distribuidor'].tolist():
            df_meta_kg = pd.concat(
                [df_meta_kg, pd.DataFrame([{'nome_distribuidor': d, 'meta_kg_dist': 0}])],
                ignore_index=True
            )
        if d not in df_meta_rs['nome_distribuidor'].tolist():
            df_meta_rs = pd.concat(
                [df_meta_rs, pd.DataFrame([{'nome_distribuidor': d, 'meta_r_dist': 0}])],
                ignore_index=True
            )

    df_meta_kg['nome_distribuidor'] = df_meta_kg['nome_distribuidor'].astype(str)
    df_meta_rs['nome_distribuidor'] = df_meta_rs['nome_distribuidor'].astype(str)
    return df_meta_kg, df_meta_rs

def calcular_comissoes_mensais(df_fatur, selected_dist, selected_produtos, pct1, pct2, pct3, selected_ano):
    resultados = []
    for mes in range(1, 13):
        # Faturamento atual para o mês
        df_curr = df_fatur[
            (df_fatur['nome_distribuidor'].isin(selected_dist)) &
            (df_fatur['ano'] == selected_ano) &
            (df_fatur['mes'] == mes)
        ].copy()
        if selected_produtos:
            df_curr = df_curr[df_curr['codigo_produto'].isin(selected_produtos)]
        agrup = ['nome_distribuidor', 'codigo_produto']
        df_current = df_curr.groupby(agrup).agg(
            Total_Kg_Mes=('total kg', 'sum'),
            Faturamento_Mes=('total df', 'sum')
        ).reset_index()
        df_current['Preco_Kg_Mes'] = df_current.apply(
            lambda r: r['Faturamento_Mes'] / r['Total_Kg_Mes'] if r['Total_Kg_Mes'] > 0 else 0,
            axis=1
        )

        # Faturamento ano anterior, mesmo mês
        df_prev = df_fatur[
            (df_fatur['nome_distribuidor'].isin(selected_dist)) &
            (df_fatur['ano'] == (selected_ano - 1)) &
            (df_fatur['mes'] == mes)
        ].copy()
        if selected_produtos:
            df_prev = df_prev[df_prev['codigo_produto'].isin(selected_produtos)]
        df_prev_group = df_prev.groupby(agrup).agg(
            Total_Kg_Ant=('total kg', 'sum'),
            Faturamento_Ant=('total df', 'sum')
        ).reset_index()
        df_prev_group['Preco_Kg_Ant'] = df_prev_group.apply(
            lambda r: r['Faturamento_Ant'] / r['Total_Kg_Ant'] if r['Total_Kg_Ant'] > 0 else 0,
            axis=1
        )

        # Merge faturamento atual x anterior
        df_merge = pd.merge(
            df_current, df_prev_group,
            on=['nome_distribuidor', 'codigo_produto'], how='left'
        ).fillna({'Total_Kg_Ant': 0, 'Faturamento_Ant': 0, 'Preco_Kg_Ant': 0})

        # Carregar metas por distribuidor (sem produto)
        selected_dist_upper = [d.strip().upper() for d in selected_dist]
        df_meta_kg_dist, df_meta_rs_dist = load_meta_filtered(selected_dist_upper, selected_ano, mes)

        df_merge['nome_distribuidor_up'] = df_merge['nome_distribuidor'].str.strip().str.upper()
        df_merge = pd.merge(
            df_merge,
            df_meta_kg_dist[['nome_distribuidor', 'meta_kg_dist']].rename(columns={'nome_distribuidor': 'nome_distribuidor_up'}),
            on='nome_distribuidor_up', how='left'
        ).fillna({'meta_kg_dist': 0})
        df_merge = pd.merge(
            df_merge,
            df_meta_rs_dist[['nome_distribuidor', 'meta_r_dist']].rename(columns={'nome_distribuidor': 'nome_distribuidor_up'}),
            on='nome_distribuidor_up', how='left'
        ).fillna({'meta_r_dist': 0})

        # Número de SKUs por distribuidor
        df_merge['n_produtos'] = df_merge.groupby('nome_distribuidor_up')['codigo_produto'] \
                                         .transform('nunique').astype(int)

        # Dividir meta entre SKUs
        df_merge['meta_kg'] = df_merge.apply(
            lambda r: (r['meta_kg_dist'] / r['n_produtos']) if r['n_produtos'] > 0 else 0,
            axis=1
        )
        df_merge['meta_r$'] = df_merge.apply(
            lambda r: (r['meta_r_dist'] / r['n_produtos']) if r['n_produtos'] > 0 else 0,
            axis=1
        )

        # Calcular deltas
        df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']
        df_merge['Delta_R']  = df_merge['Faturamento_Mes'] - df_merge['Faturamento_Ant']

        # Calcular faixas de KG
        def calcular_faixas(row):
            total = row['Total_Kg_Mes']
            prev = row['Total_Kg_Ant']
            meta = row['meta_kg']
            if meta == 0:
                kg_t1 = min(total, prev)
                kg_t2 = 0
                kg_t3 = max(total - prev, 0)
            else:
                kg_t1 = min(total, prev)
                kg_t2 = min(max(total - prev, 0), max(meta - prev, 0))
                kg_t3 = max(total - meta, 0)
            return pd.Series({'Kg_T1': kg_t1, 'Kg_T2': kg_t2, 'Kg_T3': kg_t3})

        faixas = df_merge.apply(calcular_faixas, axis=1)
        df_merge = pd.concat([df_merge, faixas], axis=1)

        # Valor em R$ por faixa
        df_merge['Val_T1'] = df_merge['Kg_T1'] * df_merge['Preco_Kg_Mes']
        df_merge['Val_T2'] = df_merge['Kg_T2'] * df_merge['Preco_Kg_Mes']
        df_merge['Val_T3'] = df_merge['Kg_T3'] * df_merge['Preco_Kg_Mes']

        # Comissão por faixa
        df_merge['Com_T1'] = df_merge['Val_T1'] * (pct1 / 100)
        df_merge['Com_T2'] = df_merge['Val_T2'] * (pct2 / 100)
        df_merge['Com_T3'] = df_merge['Val_T3'] * (pct3 / 100)

        df_merge['Comissao_R$'] = df_merge['Com_T1'] + df_merge['Com_T2'] + df_merge['Com_T3']

        # Resumo anual (apenas para gráfico)
        df_sum = df_merge.groupby('nome_distribuidor')['Comissao_R$'].sum().reset_index()
        df_sum['mes'] = mes
        resultados.append(df_sum)

    df_annual = pd.concat(resultados, ignore_index=True)
    return df_annual

def main():
    st.title("📊 Calculadora de Comissões por KG com Metas e Incrementos")
    st.markdown("Selecione as opções à esquerda e clique em **Calcular** para gerar o relatório de comissões mensais e anual.\n")

    st.sidebar.header("📁 Importar dados e selecionar filtros")
    uploaded_file = st.sidebar.file_uploader("Arquivo Excel da base de faturamento", type=["xlsx", "xls"])

    # Inicializar variáveis antes de usar no form, para evitar UnboundLocalError
    distribuidores, anos, meses, produtos = [], [], [], []
    selected_dist, selected_ano, selected_mes, selected_produtos = [], None, None, []

    if uploaded_file:
        with st.spinner("Carregando arquivo..."):
            try:
                df_raw = pd.read_excel(uploaded_file)
                df_fatur = load_faturamento(df_raw)
            except Exception as e:
                st.sidebar.error(f"Falha ao ler o arquivo Excel: {e}")
                df_fatur = None

        if df_fatur is not None:
            distribuidores = sorted(df_fatur['nome_distribuidor'].dropna().unique())
            anos = sorted(df_fatur['ano'].dropna().astype(int).unique())
            meses = list(range(1, 13))
            produtos = sorted(df_fatur['codigo_produto'].dropna().unique())

        # Formulário de filtros
        with st.sidebar.form(key="filtros_form"):
            st.subheader("📋 Filtros")
            dist_selecionados = st.multiselect("Distribuidores", distribuidores, help="Selecione distribuidores")
            ano_selecionado = st.selectbox("Ano de análise", anos, index=len(anos)-1 if anos else 0) if anos else None
            mes_selecionado = st.selectbox(
                "Mês de análise", 
                meses, 
                format_func=lambda x: f"{x:02d}",
                index=datetime.now().month-1
            ) if meses else None
            prod_selecionados = st.multiselect("Produtos (código)", produtos, help="Selecione produtos")
            
            st.markdown("---")
            st.subheader("⚙️ Configuração de Comissões")
            pct1 = st.number_input("% Até volume do ano anterior", value=2.0, format="%.1f")
            pct2 = st.number_input("% Volume entre ano anterior e meta", value=4.0, format="%.1f")
            pct3 = st.number_input("% Acima da meta", value=6.0, format="%.1f")
            st.markdown("---")
            
            btn_calcular = st.form_submit_button("🔍 Calcular")

        if btn_calcular:
            if df_fatur is None:
                st.error("Envie o arquivo Excel antes de calcular.")
            else:
                selected_dist = dist_selecionados
                selected_ano = ano_selecionado
                selected_mes = mes_selecionado
                selected_produtos = prod_selecionados

                with st.spinner("Processando dados..."):
                    # -------------------------------
                    # Cálculo para o mês selecionado
                    # -------------------------------
                    df_curr = df_fatur[
                        (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                        (df_fatur['ano'] == selected_ano) &
                        (df_fatur['mes'] == selected_mes)
                    ].copy()
                    if selected_produtos:
                        df_curr = df_curr[df_curr['codigo_produto'].isin(selected_produtos)]
                    agrup = ['nome_distribuidor', 'codigo_produto']
                    df_current = df_curr.groupby(agrup).agg(
                        Total_Kg_Mes=('total kg', 'sum'),
                        Faturamento_Mes=('total df', 'sum')
                    ).reset_index()
                    df_current['Preco_Kg_Mes'] = df_current.apply(
                        lambda r: r['Faturamento_Mes'] / r['Total_Kg_Mes'] if r['Total_Kg_Mes'] > 0 else 0,
                        axis=1
                    )

                    df_prev = df_fatur[
                        (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                        (df_fatur['ano'] == (selected_ano - 1)) &
                        (df_fatur['mes'] == selected_mes)
                    ].copy()
                    if selected_produtos:
                        df_prev = df_prev[df_prev['codigo_produto'].isin(selected_produtos)]
                    df_prev_group = df_prev.groupby(agrup).agg(
                        Total_Kg_Ant=('total kg', 'sum'),
                        Faturamento_Ant=('total df', 'sum')
                    ).reset_index()
                    df_prev_group['Preco_Kg_Ant'] = df_prev_group.apply(
                        lambda r: r['Faturamento_Ant'] / r['Total_Kg_Ant'] if r['Total_Kg_Ant'] > 0 else 0,
                        axis=1
                    )

                    df_merge = pd.merge(
                        df_current, df_prev_group,
                        on=['nome_distribuidor', 'codigo_produto'], how='left'
                    ).fillna({'Total_Kg_Ant': 0, 'Faturamento_Ant': 0, 'Preco_Kg_Ant': 0})

                    df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']
                    df_merge['Delta_R']  = df_merge['Faturamento_Mes'] - df_merge['Faturamento_Ant']

                    df_meta_kg_dist, df_meta_rs_dist = load_meta_filtered(
                        selected_dist, selected_ano, selected_mes
                    )

                    df_merge['nome_distribuidor_up'] = df_merge['nome_distribuidor'].str.strip().str.upper()
                    df_merge = pd.merge(
                        df_merge,
                        df_meta_kg_dist[['nome_distribuidor', 'meta_kg_dist']].rename(columns={'nome_distribuidor': 'nome_distribuidor_up'}),
                        on='nome_distribuidor_up', how='left'
                    ).fillna({'meta_kg_dist': 0})
                    df_merge = pd.merge(
                        df_merge,
                        df_meta_rs_dist[['nome_distribuidor', 'meta_r_dist']].rename(columns={'nome_distribuidor': 'nome_distribuidor_up'}),
                        on='nome_distribuidor_up', how='left'
                    ).fillna({'meta_r_dist': 0})

                    df_merge['n_produtos'] = df_merge.groupby('nome_distribuidor_up')['codigo_produto'] \
                                                     .transform('nunique').astype(int)

                    df_merge['meta_kg'] = df_merge.apply(
                        lambda r: (r['meta_kg_dist'] / r['n_produtos']) if r['n_produtos'] > 0 else 0,
                        axis=1
                    )
                    df_merge['meta_r$'] = df_merge.apply(
                        lambda r: (r['meta_r_dist'] / r['n_produtos']) if r['n_produtos'] > 0 else 0,
                        axis=1
                    )

                    def calcular_faixas(row):
                        total = row['Total_Kg_Mes']
                        prev = row['Total_Kg_Ant']
                        meta = row['meta_kg']
                        if meta == 0:
                            kg_t1 = min(total, prev)
                            kg_t2 = 0
                            kg_t3 = max(total - prev, 0)
                        else:
                            kg_t1 = min(total, prev)
                            kg_t2 = min(max(total - prev, 0), max(meta - prev, 0))
                            kg_t3 = max(total - meta, 0)
                        return pd.Series({'Kg_T1': kg_t1, 'Kg_T2': kg_t2, 'Kg_T3': kg_t3})

                    faixas = df_merge.apply(calcular_faixas, axis=1)
                    df_merge = pd.concat([df_merge, faixas], axis=1)

                    df_merge['Val_T1'] = df_merge['Kg_T1'] * df_merge['Preco_Kg_Mes']
                    df_merge['Val_T2'] = df_merge['Kg_T2'] * df_merge['Preco_Kg_Mes']
                    df_merge['Val_T3'] = df_merge['Kg_T3'] * df_merge['Preco_Kg_Mes']

                    df_merge['Com_T1'] = df_merge['Val_T1'] * (pct1 / 100)
                    df_merge['Com_T2'] = df_merge['Val_T2'] * (pct2 / 100)
                    df_merge['Com_T3'] = df_merge['Val_T3'] * (pct3 / 100)

                    df_merge['Comissao_R$'] = df_merge['Com_T1'] + df_merge['Com_T2'] + df_merge['Com_T3']

                    df_display = df_merge.copy()
                    # Formatação de colunas de KG sem casas decimais
                    df_display['Distribuidor'] = df_display['nome_distribuidor']
                    df_display['Produto'] = df_display['codigo_produto']
                    df_display['Kg Ano Anterior'] = df_display['Total_Kg_Ant'].apply(lambda x: f"{x:,.0f}")
                    df_display['Meta Kg (dividido)'] = df_display['meta_kg'].apply(lambda x: f"{x:,.0f}")
                    df_display['Kg Mês'] = df_display['Total_Kg_Mes'].apply(lambda x: f"{x:,.0f}")
                    df_display['Δ Kg'] = df_display['Delta_Kg'].apply(lambda x: f"{x:,.0f}")
                    # Agora “Kg Entre Ano Ant. e Meta” = Meta Kg (dividido) – Kg Ano Anterior
                    df_display['Kg Entre Ano Ant. e Meta'] = df_display.apply(
                        lambda r: f"{max(r['meta_kg'] - r['Total_Kg_Ant'], 0):,.0f}", axis=1
                    )
                    df_display['Kg Até Ano Anterior'] = df_display['Kg_T1'].apply(lambda x: f"{x:,.0f}")
                    df_display['Kg Acima da Meta'] = df_display['Kg_T3'].apply(lambda x: f"{x:,.0f}")
                    df_display['Preço/kg Mês (R$)'] = df_display['Preco_Kg_Mes'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Valor Até Ano Anterior (R$)'] = df_display['Val_T1'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Valor Faixa Meta (R$)'] = df_display['Val_T2'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Valor Acima Meta (R$)'] = df_display['Val_T3'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Comissão T1 (R$)'] = df_display['Com_T1'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Comissão T2 (R$)'] = df_display['Com_T2'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Comissão T3 (R$)'] = df_display['Com_T3'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )
                    df_display['Comissão Total (R$)'] = df_display['Comissao_R$'].apply(
                        lambda x: f"R$ {x:,.2f}"
                    )

                    df_display = df_display[[
                        'Distribuidor', 'Produto', 'Kg Ano Anterior', 'Meta Kg (dividido)', 'Kg Mês', 'Δ Kg',
                        'Kg Entre Ano Ant. e Meta', 'Kg Até Ano Anterior', 'Kg Acima da Meta',
                        'Preço/kg Mês (R$)', 'Valor Até Ano Anterior (R$)',
                        'Valor Faixa Meta (R$)', 'Valor Acima Meta (R$)',
                        'Comissão T1 (R$)', 'Comissão T2 (R$)',
                        'Comissão T3 (R$)', 'Comissão Total (R$)'
                    ]]

                # Exibição dos resultados do mês selecionado
                if df_display.empty:
                    st.warning("Nenhum dado encontrado para os filtros selecionados.")
                else:
                    st.subheader(f"📅 Resultados – {selected_mes:02d}/{selected_ano}")

                    # Exibe a tabela com cabeçalhos normais
                    st.dataframe(df_display)

                    # ----------------------------
                    # Legenda dos Cálculos
                    # ----------------------------
                    st.markdown("#### 📝 Legenda das Colunas")
                    st.markdown(
                        """
                        - **Kg Ano Anterior**: soma de `total kg` no mesmo mês do ano anterior.  
                        - **Meta Kg (dividido)**: meta de Kg do distribuidor naquele mês (tirada da tabela de metas), dividida pelo número de SKUs ativos.  
                        - **Kg Mês**: soma de `total kg` para o mês selecionado.  
                        - **Δ Kg**: diferença entre `Kg Mês` e `Kg Ano Anterior`.  
                        - **Kg Entre Ano Ant. e Meta**: `Meta Kg (dividido) – Kg Ano Anterior` (ou zero, se o resultado for negativo).  
                        - **Kg Até Ano Anterior (Kg_T1)**: parte do volume do mês que coincide com o volume até o ano anterior (ou seja, mínimo entre total atual e total do ano anterior).  
                        - **Kg Acima da Meta (Kg_T3)**: volume que excede a meta individual.  
                        - **Preço/kg Mês (R$)**: `Faturamento_Mes` dividido por `Total_Kg_Mes` (quando `Total_Kg_Mes > 0`), formatado em real.  
                        - **Valor Até Ano Anterior (R$)**: `Kg_T1 * Preço/kg Mês`.  
                        - **Valor Faixa Meta (R$)**: `Kg_T2 * Preço/kg Mês`.  
                        - **Valor Acima Meta (R$)**: `Kg_T3 * Preço/kg Mês`.  
                        - **Comissão T1 (R$)**: `Valor Até Ano Anterior * (pct1 / 100)`.  
                        - **Comissão T2 (R$)**: `Valor Faixa Meta * (pct2 / 100)`.  
                        - **Comissão T3 (R$)**: `Valor Acima Meta * (pct3 / 100)`.  
                        - **Comissão Total (R$)**: soma de `Comissão T1 + Comissão T2 + Comissão T3`.  
                        """
                    )

                    # Totais Consolidados (Mês Selecionado)
                    st.markdown("---")
                    st.markdown("**Totais Consolidados (Mês Selecionado)**")
                    totais_merge = df_merge.groupby('nome_distribuidor').agg(
                        Total_Kg_Ant=('Total_Kg_Ant', 'sum'),
                        Total_Kg_Mes=('Total_Kg_Mes', 'sum'),
                        Sum_Meta_Kg=('meta_kg', 'sum'),
                        Total_Fat_Mes=('Faturamento_Mes', 'sum'),
                        Kg_T1_Total=('Kg_T1', 'sum'),
                        Kg_T2_Total=('Kg_T2', 'sum'),
                        Kg_T3_Total=('Kg_T3', 'sum'),
                        Val_T1_Total=('Val_T1', 'sum'),
                        Val_T2_Total=('Val_T2', 'sum'),
                        Val_T3_Total=('Val_T3', 'sum'),
                        Com_T1_Total=('Com_T1', 'sum'),
                        Com_T2_Total=('Com_T2', 'sum'),
                        Com_T3_Total=('Com_T3', 'sum'),
                        Comissao_Total=('Comissao_R$', 'sum')
                    ).reset_index().rename(columns={'nome_distribuidor': 'Distribuidor'})

                    totais_merge['Preco_Medio_Kg'] = totais_merge.apply(
                        lambda r: (r['Total_Fat_Mes'] / r['Total_Kg_Mes']) if r['Total_Kg_Mes'] > 0 else 0,
                        axis=1
                    )

                    totais_merge['Kg Ano Anterior'] = totais_merge['Total_Kg_Ant'].apply(lambda x: f"{x:,.0f}")
                    totais_merge['Kg Mês'] = totais_merge['Total_Kg_Mes'].apply(lambda x: f"{x:,.0f}")
                    totais_merge['Meta Kg (dividido)'] = totais_merge['Sum_Meta_Kg'].apply(lambda x: f"{x:,.0f}")
                    totais_merge['Kg Entre Ano Ant. e Meta'] = totais_merge.apply(
                        lambda r: f"{max(r['Sum_Meta_Kg'] - r['Total_Kg_Ant'], 0):,.0f}", axis=1
                    )
                    totais_merge['Preço Médio (R$/Kg)'] = totais_merge['Preco_Medio_Kg'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Kg Até Ano Anterior'] = totais_merge['Kg_T1_Total'].apply(lambda x: f"{x:,.0f}")
                    totais_merge['Kg Acima da Meta'] = totais_merge['Kg_T3_Total'].apply(lambda x: f"{x:,.0f}")
                    totais_merge['Valor Até Ano Anterior (R$)'] = totais_merge['Val_T1_Total'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Valor Faixa Meta (R$)'] = totais_merge['Val_T2_Total'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Valor Acima Meta (R$)'] = totais_merge['Val_T3_Total'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Comissão T1 (R$)'] = totais_merge['Com_T1_Total'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Comissão T2 (R$)'] = totais_merge['Com_T2_Total'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Comissão T3 (R$)'] = totais_merge['Com_T3_Total'].apply(lambda x: f"R$ {x:,.2f}")
                    totais_merge['Comissão Total (R$)'] = totais_merge['Comissao_Total'].apply(lambda x: f"R$ {x:,.2f}")

                    totais_exib = totais_merge[[ 
                        'Distribuidor',
                        'Kg Ano Anterior', 'Kg Mês', 'Meta Kg (dividido)', 'Kg Entre Ano Ant. e Meta',
                        'Preço Médio (R$/Kg)', 'Kg Até Ano Anterior', 'Kg Acima da Meta',
                        'Valor Até Ano Anterior (R$)', 'Valor Faixa Meta (R$)', 'Valor Acima Meta (R$)',
                        'Comissão T1 (R$)', 'Comissão T2 (R$)', 'Comissão T3 (R$)', 'Comissão Total (R$)'
                    ]]

                    st.write(totais_exib)

                    # -------------------------------------------------------
                    # Gráfico de barras do mês selecionado (Altair + labels)
                    # -------------------------------------------------------
                    st.markdown("**Gráfico de Comissões por Distribuidor (Mês Selecionado)**")
                    df_graf_mes = totais_merge[['Distribuidor', 'Comissao_Total']].copy()
                    df_graf_mes['Comissao_Num'] = totais_merge['Comissao_Total'].replace(r'[R\$,]', '', regex=True).astype(float)

                    base_mes = alt.Chart(df_graf_mes).encode(
                        x=alt.X('Distribuidor:N', sort=None, title='Distribuidor'),
                        y=alt.Y('Comissao_Num:Q', title='Comissão (R$)'),
                        tooltip=[alt.Tooltip('Comissao_Num:Q', title='Comissão (R$)', format=',.2f')]
                    )

                    bars_mes = base_mes.mark_bar().encode(color=alt.Color('Distribuidor:N', legend=None))
                    text_mes = base_mes.mark_text(dy=-5, size=12).encode(text=alt.Text('Comissao_Num:Q', format=',.2f'))

                    chart_mes = (bars_mes + text_mes).properties(width='container', height=400)
                    st.altair_chart(chart_mes, use_container_width=True)

                    # -------------------------------------------------------
                    # Gráfico anual (Altair + labels)
                    # -------------------------------------------------------
                    st.markdown("---")
                    st.markdown("**Gráfico Anual de Comissões por Mês e Distribuidor**")

                    df_annual = calcular_comissoes_mensais(
                        df_fatur, selected_dist, selected_produtos, pct1, pct2, pct3, selected_ano
                    )
                    df_annual['mes_str'] = df_annual['mes'].apply(lambda x: f"{x:02d}")
                    df_annual.rename(columns={'nome_distribuidor': 'Distribuidor', 'Comissao_R$': 'Comissao_Num'}, inplace=True)

                    base_annual = alt.Chart(df_annual).encode(
                        x=alt.X('mes_str:O', title='Mês'),
                        y=alt.Y('Comissao_Num:Q', stack='zero', title='Comissão (R$)'),
                        color=alt.Color('Distribuidor:N', title='Distribuidor'),
                        tooltip=[
                            alt.Tooltip('Distribuidor:N'),
                            alt.Tooltip('mes_str:O', title='Mês'),
                            alt.Tooltip('Comissao_Num:Q', title='Comissão (R$)', format=',.2f')
                        ]
                    )

                    bars_annual = base_annual.mark_bar()
                    text_annual = base_annual.mark_text(size=10, dy=0).encode(
                        text=alt.Text('Comissao_Num:Q', format=',.2f'),
                        y=alt.Y('Comissao_Num:Q', stack='center')
                    )

                    chart_annual = (bars_annual + text_annual).properties(width='container', height=400)
                    st.altair_chart(chart_annual, use_container_width=True)

        else:
            st.sidebar.info("Faça upload do Excel, escolha filtros e clique em 'Calcular'.")
    else:
        st.sidebar.info("Por favor, carregue um arquivo Excel para começar.")

if __name__ == "__main__":
    main()
