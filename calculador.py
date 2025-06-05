import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# -------------------------------------------------------
# Configurar a página (primeiro comando do Streamlit)
# -------------------------------------------------------
st.set_page_config(page_title="Calculadora de Comissões por KG", layout="wide")

# -------------------------------------------------------
# Funções auxiliares para carregar faturamento
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_faturamento(df_fatur):
    df = df_fatur.copy()
    # Normaliza nomes de colunas
    df.columns = df.columns.str.strip().str.lower()
    # Converte data de emissão
    df['emissao'] = pd.to_datetime(df['emissao'], dayfirst=True, errors='coerce')
    # Converte numéricos
    df['unid_faturado'] = pd.to_numeric(df['unid_faturado'], errors='coerce').fillna(0)
    df['total df']     = pd.to_numeric(df['total df'], errors='coerce').fillna(0.0)
    df['total kg']     = pd.to_numeric(df['total kg'], errors='coerce').fillna(0.0)
    # Extrai ano e mês
    df['ano'] = df['emissao'].dt.year
    df['mes'] = df['emissao'].dt.month
    # Garante strings
    df['nome_distribuidor'] = df['nome_distribuidor'].astype(str)
    df['codigo_produto']    = df['codigo_produto'].astype(str)
    return df

# -------------------------------------------------------
# Função para carregar e “desempilhar” metas diárias
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metas(metas_file):
    """
    Lê um arquivo de metas que pode ser CSV (sep=";") ou Excel.
    Cada aba (ou a tabela única, no caso de CSV) deve conter:
      NOME DISTRIBUIDOR | COD PROD | 01/01/2025 | 02/01/2025 | ...
    Retorna um DataFrame com colunas:
      ['nome_distribuidor', 'codigo_produto', 'data_dia', 'meta_kg_dia']
      Onde meta_kg_dia é float arredondado a duas casas decimais.
    """
    # Detecta extensão e carrega adequadamente
    name = metas_file.name.lower()
    if name.endswith('.csv'):
        # CSV usa ";" como separador, e no seu caso o ponto já é o separador decimal
        df_raw = pd.read_csv(metas_file, sep=';', dtype=str)
        sheets = {'sheet1': df_raw}
    else:
        sheets = pd.read_excel(metas_file, sheet_name=None, dtype=str)

    lista_metas = []
    for sheet_name, df_raw in sheets.items():
        df = df_raw.copy()
        # Normaliza colunas
        df.columns = df.columns.str.strip()
        # Verifica colunas mínimas
        if 'NOME DISTRIBUIDOR' not in df.columns or 'COD PROD' not in df.columns:
            continue

        colunas = df.columns.tolist()
        idx_inicio_datas = colunas.index('COD PROD') + 1
        datas_cols = colunas[idx_inicio_datas:]

        # “Desempilha” datas
        melt = df.melt(
            id_vars=['NOME DISTRIBUIDOR', 'COD PROD'],
            value_vars=datas_cols,
            var_name='data_dia',
            value_name='meta_kg_dia'
        )
        # Converte data_dia para datetime
        melt['data_dia'] = pd.to_datetime(melt['data_dia'], dayfirst=True, errors='coerce')

        # — Limpeza e conversão de meta_kg_dia para float —
        # 1) Passamos para string
        # 2) Tiramos as vírgulas que forem apenas separadores de milhar (",")
        # 3) Mantemos o ponto como separador decimal (no seu caso, o CSV já vem assim)
        # 4) Convertemos para float
        melt['meta_kg_dia'] = (
            melt['meta_kg_dia']
            .astype(str)
            .str.replace(',', '', regex=False)      # remove vírgulas de milhar, se houver
        )
        melt['meta_kg_dia'] = pd.to_numeric(melt['meta_kg_dia'], errors='coerce').fillna(0.0)
        # Arredonda a duas casas decimais
        melt['meta_kg_dia'] = melt['meta_kg_dia'].round(2)

        # Padroniza nomes
        melt.rename(columns={'NOME DISTRIBUIDOR': 'nome_distribuidor',
                             'COD PROD': 'codigo_produto'}, inplace=True)

        lista_metas.append(
            melt[['nome_distribuidor', 'codigo_produto', 'data_dia', 'meta_kg_dia']]
        )

    if not lista_metas:
        return pd.DataFrame(columns=['nome_distribuidor', 'codigo_produto', 'data_dia', 'meta_kg_dia'])

    df_metas = pd.concat(lista_metas, ignore_index=True)
    df_metas = df_metas[df_metas['data_dia'].notna()].copy()
    return df_metas

# -------------------------------------------------------
# Função para agregar metas diárias em metas mensais
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def aggregate_metas_mensais(df_metas):
    """
    Recebe DataFrame de metas diárias:
      ['nome_distribuidor','codigo_produto','data_dia','meta_kg_dia']
    Retorna DataFrame com metas mensais:
      ['nome_distribuidor','codigo_produto','ano','mes','meta_kg_mes']
      Onde 'meta_kg_mes' é soma dos valores arredondados a duas casas.
    """
    df = df_metas.copy()
    df['ano'] = df['data_dia'].dt.year
    df['mes'] = df['data_dia'].dt.month
    df_mes = (
        df
        .groupby(['nome_distribuidor', 'codigo_produto', 'ano', 'mes'], as_index=False)
        .agg(meta_kg_mes=('meta_kg_dia', 'sum'))
    )
    # Garante strings
    df_mes['nome_distribuidor'] = df_mes['nome_distribuidor'].astype(str)
    df_mes['codigo_produto']    = df_mes['codigo_produto'].astype(str)
    # Arredonda novamente (caso a soma precise ajuste)
    df_mes['meta_kg_mes'] = df_mes['meta_kg_mes'].round(2)
    return df_mes

# -------------------------------------------------------
# Função de cálculo de comissões mensais (sem DB)
# -------------------------------------------------------
def calcular_comissoes_mensais(
    df_fatur, df_meta_mensal, selected_dist,
    selected_produtos, pct1, pct2, pct3, selected_ano
):
    resultados = []
    for mes in range(1, 13):
        # 1) Filtra faturamento do ano selecionado / mês / distribuidores / produtos
        df_curr = df_fatur[
            (df_fatur['nome_distribuidor'].isin(selected_dist)) &
            (df_fatur['ano'] == selected_ano) &
            (df_fatur['mes'] == mes)
        ].copy()
        if selected_produtos:
            df_curr = df_curr[df_curr['codigo_produto'].isin(selected_produtos)]

        agrup = ['nome_distribuidor', 'codigo_produto']
        df_current = (
            df_curr
            .groupby(agrup, as_index=False)
            .agg(
                Total_Kg_Mes=('total kg', 'sum'),
                Faturamento_Mes=('total df', 'sum')
            )
        )
        df_current['Preco_Kg_Mes'] = df_current.apply(
            lambda r: (r['Faturamento_Mes'] / r['Total_Kg_Mes']) if r['Total_Kg_Mes'] > 0 else 0,
            axis=1
        )

        # 2) Filtra faturamento do ano anterior / mesmo mês (para Δ)
        df_prev = df_fatur[
            (df_fatur['nome_distribuidor'].isin(selected_dist)) &
            (df_fatur['ano'] == (selected_ano - 1)) &
            (df_fatur['mes'] == mes)
        ].copy()
        if selected_produtos:
            df_prev = df_prev[df_prev['codigo_produto'].isin(selected_produtos)]
        df_prev_group = (
            df_prev
            .groupby(agrup, as_index=False)
            .agg(
                Total_Kg_Ant=('total kg', 'sum'),
                Faturamento_Ant=('total df', 'sum')
            )
        )
        df_prev_group['Preco_Kg_Ant'] = df_prev_group.apply(
            lambda r: (r['Faturamento_Ant'] / r['Total_Kg_Ant']) if r['Total_Kg_Ant'] > 0 else 0,
            axis=1
        )

        # 3) Merge entre corrente e anterior
        df_merge = pd.merge(
            df_current, df_prev_group,
            on=['nome_distribuidor', 'codigo_produto'],
            how='left'
        ).fillna({'Total_Kg_Ant': 0, 'Faturamento_Ant': 0, 'Preco_Kg_Ant': 0})

        # 4) Calcula Δ
        df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']
        df_merge['Delta_R']  = df_merge['Faturamento_Mes'] - df_merge['Faturamento_Ant']

        # 5) Agrega meta mensal (ano=selected_ano, mes=mes)
        df_meta_mes_corrente = df_meta_mensal[
            (df_meta_mensal['ano'] == selected_ano) &
            (df_meta_mensal['mes'] == mes)
        ][['nome_distribuidor', 'codigo_produto', 'meta_kg_mes']].copy()
        df_merge = pd.merge(
            df_merge,
            df_meta_mes_corrente,
            on=['nome_distribuidor', 'codigo_produto'],
            how='left'
        ).fillna({'meta_kg_mes': 0.0})
        df_merge.rename(columns={'meta_kg_mes': 'meta_kg'}, inplace=True)

        # 6) Calcula faixas Kg_T1, Kg_T2, Kg_T3
        def calcular_faixas(row):
            total = row['Total_Kg_Mes']
            prev  = row['Total_Kg_Ant']
            meta  = row['meta_kg']
            if prev >= meta:
                kg_t1 = min(total, prev)
                kg_t2 = 0
                kg_t3 = max(total - prev, 0)
            else:
                if total <= prev:
                    kg_t1 = total
                    kg_t2 = 0
                    kg_t3 = 0
                else:
                    kg_t1 = prev
                    kg_t2 = min(total - prev, max(meta - prev, 0))
                    kg_t3 = max(total - meta, 0)
            return pd.Series({'Kg_T1': kg_t1, 'Kg_T2': kg_t2, 'Kg_T3': kg_t3})

        if df_merge.empty:
            df_merge[['Kg_T1', 'Kg_T2', 'Kg_T3']] = 0, 0, 0
        else:
            faixas = df_merge.apply(calcular_faixas, axis=1)
            faixas.columns = [c.strip() for c in faixas.columns]
            df_merge[['Kg_T1', 'Kg_T2', 'Kg_T3']] = faixas[['Kg_T1', 'Kg_T2', 'Kg_T3']]

        # 7) Calcula valores e comissões
        df_merge['Val_T1'] = df_merge['Kg_T1'] * df_merge['Preco_Kg_Mes']
        df_merge['Val_T2'] = df_merge['Kg_T2'] * df_merge['Preco_Kg_Mes']
        df_merge['Val_T3'] = df_merge['Kg_T3'] * df_merge['Preco_Kg_Mes']

        df_merge['Com_T1'] = df_merge['Val_T1'] * (pct1 / 100)
        df_merge['Com_T2'] = df_merge['Val_T2'] * (pct2 / 100)
        df_merge['Com_T3'] = df_merge['Val_T3'] * (pct3 / 100)
        df_merge['Comissao_R$'] = df_merge['Com_T1'] + df_merge['Com_T2'] + df_merge['Com_T3']

        # 8) Agrupa apenas comissão total por distribuidor
        df_sum = (
            df_merge
            .groupby('nome_distribuidor', as_index=False)
            .agg(**{'Comissao_R$': ('Comissao_R$', 'sum')})
        )
        df_sum['mes'] = mes
        resultados.append(df_sum)

    if resultados:
        df_annual = pd.concat(resultados, ignore_index=True)
    else:
        df_annual = pd.DataFrame(columns=['nome_distribuidor', 'Comissao_R$', 'mes'])
    return df_annual

# -------------------------------------------------------
# Fluxo principal
# -------------------------------------------------------
def main():
    st.title("📊 Calculadora de Comissões por KG")

    st.markdown("""
    Este aplicativo calcula comissões mensais e anual com base em:
    1. **Base de faturamento** (Excel carregado pelo usuário);
    2. **Metas diárias** (CSV ou Excel), onde cada célula representa a meta de KG daquele dia, por distribuidor e produto.
    """)

    st.sidebar.header("📁 Importar dados")
    # 1) Uploader da base de faturamento
    uploaded_fat = st.sidebar.file_uploader(
        "1) Carregue aqui o Excel da base de faturamento",
        type=["xlsx", "xls"], key="fat"
    )
    # 2) Uploader das metas diárias (CSV ou Excel)
    uploaded_meta = st.sidebar.file_uploader(
        "2) Carregue aqui o arquivo de metas diárias (CSV ou Excel)",
        type=["csv", "xlsx", "xls"], key="meta"
    )

    df_fatur = None
    df_metas_diarias = None
    df_meta_mensal  = None

    # Se o usuário carregou o faturamento, converte para df
    if uploaded_fat:
        try:
            df_raw = pd.read_excel(uploaded_fat)
            df_fatur = load_faturamento(df_raw)
        except Exception as e:
            st.sidebar.error(f"Falha ao ler Faturamento: {e}")
            df_fatur = None

    # Se o usuário carregou o arquivo de metas, “desempilha” e agrega mensal
    if uploaded_meta:
        try:
            df_metas_diarias = load_metas(uploaded_meta)
            df_meta_mensal  = aggregate_metas_mensais(df_metas_diarias)
        except Exception as e:
            st.sidebar.error(f"Falha ao ler Metas diárias: {e}")
            df_metas_diarias = None
            df_meta_mensal   = None

    # Se faturamento e metas estiverem OK, preenche as variáveis de filtro
    distribuidores, anos, meses, produtos = [], [], [], []
    if df_fatur is not None:
        distribuidores = sorted(df_fatur['nome_distribuidor'].dropna().unique())
        anos = sorted(df_fatur['ano'].dropna().astype(int).unique())
        meses = list(range(1, 13))
        produtos = sorted(df_fatur['codigo_produto'].dropna().unique())

    # ==== Sidebar: filtros e parâmetros ====
    with st.sidebar.form(key="filtros_form"):
        st.subheader("📋 Filtros de análise")
        dist_selecionados = st.multiselect("Distribuidores", distribuidores, help="Selecione distribuidores")
        ano_selecionado  = st.selectbox(
            "Ano de análise",
            anos,
            index=(len(anos)-1) if anos else 0
        ) if anos else None
        mes_selecionado = st.selectbox(
            "Mês de análise",
            meses,
            format_func=lambda x: f"{x:02d}",
            index=datetime.now().month - 1
        ) if meses else None
        prod_selecionados = st.multiselect("Produtos (código)", produtos, help="Selecione produtos")

        st.markdown("---")
        st.subheader("⚙️ Configuração de Comissões")
        pct1 = st.number_input("% Até volume do ano anterior", value=2.000, format="%.3f", step=0.001)
        pct2 = st.number_input("% Volume entre ano anterior e meta", value=4.000, format="%.3f", step=0.001)
        pct3 = st.number_input("% Acima da meta", value=6.000, format="%.3f", step=0.001)
        st.markdown("---")

        btn_calcular = st.form_submit_button("🔍 Calcular")

    # Só executa se clicar em Calcular
    if btn_calcular:
        # Verifica se tudo foi carregado
        if df_fatur is None:
            st.error("❌ Carregue o arquivo de faturamento antes de calcular.")
            return
        if df_meta_mensal is None:
            st.error("❌ Carregue o arquivo de metas diárias antes de calcular.")
            return

        selected_dist     = dist_selecionados
        selected_ano      = ano_selecionado
        selected_mes      = mes_selecionado
        selected_produtos = prod_selecionados

        # 1) Cálculo para o mês selecionado
        with st.spinner("Calculando mês selecionado..."):
            df_curr = df_fatur[
                (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                (df_fatur['ano'] == selected_ano) &
                (df_fatur['mes'] == selected_mes)
            ].copy()
            if selected_produtos:
                df_curr = df_curr[df_curr['codigo_produto'].isin(selected_produtos)]

            agrup = ['nome_distribuidor','codigo_produto']
            df_current = (
                df_curr
                .groupby(agrup, as_index=False)
                .agg(
                    Total_Kg_Mes=('total kg','sum'),
                    Faturamento_Mes=('total df','sum')
                )
            )
            df_current['Preco_Kg_Mes'] = df_current.apply(
                lambda r: (r['Faturamento_Mes']/r['Total_Kg_Mes']) if r['Total_Kg_Mes']>0 else 0,
                axis=1
            )

            df_prev = df_fatur[
                (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                (df_fatur['ano'] == (selected_ano-1)) &
                (df_fatur['mes'] == selected_mes)
            ].copy()
            if selected_produtos:
                df_prev = df_prev[df_prev['codigo_produto'].isin(selected_produtos)]
            df_prev_group = (
                df_prev
                .groupby(agrup, as_index=False)
                .agg(
                    Total_Kg_Ant=('total kg','sum'),
                    Faturamento_Ant=('total df','sum')
                )
            )
            df_prev_group['Preco_Kg_Ant'] = df_prev_group.apply(
                lambda r: (r['Faturamento_Ant']/r['Total_Kg_Ant']) if r['Total_Kg_Ant']>0 else 0,
                axis=1
            )

            df_merge = pd.merge(
                df_current, df_prev_group,
                on=['nome_distribuidor','codigo_produto'],
                how='left'
            ).fillna({'Total_Kg_Ant':0,'Faturamento_Ant':0,'Preco_Kg_Ant':0})

            df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']
            df_merge['Delta_R']  = df_merge['Faturamento_Mes'] - df_merge['Faturamento_Ant']

            df_meta_mes_corrente = df_meta_mensal[
                (df_meta_mensal['ano'] == selected_ano) &
                (df_meta_mensal['mes'] == selected_mes)
            ][['nome_distribuidor','codigo_produto','meta_kg_mes']].copy()
            df_merge = pd.merge(
                df_merge,
                df_meta_mes_corrente,
                on=['nome_distribuidor','codigo_produto'],
                how='left'
            ).fillna({'meta_kg_mes':0.0})
            df_merge.rename(columns={'meta_kg_mes':'meta_kg'}, inplace=True)

            def calcular_faixas(row):
                total = row['Total_Kg_Mes']
                prev  = row['Total_Kg_Ant']
                meta  = row['meta_kg']
                if prev >= meta:
                    kg_t1 = min(total, prev)
                    kg_t2 = 0
                    kg_t3 = max(total - prev, 0)
                else:
                    if total <= prev:
                        kg_t1 = total
                        kg_t2 = 0
                        kg_t3 = 0
                    else:
                        kg_t1 = prev
                        kg_t2 = min(total - prev, max(meta - prev, 0))
                        kg_t3 = max(total - meta, 0)
                return pd.Series({'Kg_T1':kg_t1,'Kg_T2':kg_t2,'Kg_T3':kg_t3})

            if df_merge.empty:
                df_merge[['Kg_T1','Kg_T2','Kg_T3']] = 0,0,0
            else:
                faixas = df_merge.apply(calcular_faixas, axis=1)
                faixas.columns = [c.strip() for c in faixas.columns]
                df_merge[['Kg_T1','Kg_T2','Kg_T3']] = faixas[['Kg_T1','Kg_T2','Kg_T3']]

            df_merge['Val_T1'] = df_merge['Kg_T1'] * df_merge['Preco_Kg_Mes']
            df_merge['Val_T2'] = df_merge['Kg_T2'] * df_merge['Preco_Kg_Mes']
            df_merge['Val_T3'] = df_merge['Kg_T3'] * df_merge['Preco_Kg_Mes']

            df_merge['Com_T1'] = df_merge['Val_T1'] * (pct1/100)
            df_merge['Com_T2'] = df_merge['Val_T2'] * (pct2/100)
            df_merge['Com_T3'] = df_merge['Val_T3'] * (pct3/100)
            df_merge['Comissao_R$'] = df_merge['Com_T1'] + df_merge['Com_T2'] + df_merge['Com_T3']

            df_display = df_merge.copy()
            df_display['Distribuidor'] = df_display['nome_distribuidor']
            df_display['Produto']     = df_display['codigo_produto']
            df_display['Kg Ano Anterior'] = df_display['Total_Kg_Ant'].apply(lambda x: f"{x:,.0f}")
            df_display['Meta Kg (mês)']   = df_display['meta_kg'].apply(lambda x: f"{x:,.0f}")
            df_display['Kg Mês']          = df_display['Total_Kg_Mes'].apply(lambda x: f"{x:,.0f}")
            df_display['Δ Kg']            = df_display['Delta_Kg'].apply(lambda x: f"{x:,.0f}")
            df_display['Kg Até Ano Anterior']    = df_display['Kg_T1'].apply(lambda x: f"{x:,.0f}")
            df_display['Kg Entre Ano Ant. e Meta'] = df_display.apply(
                lambda r: f"{max(r['meta_kg'] - r['Total_Kg_Ant'], 0):,.0f}", axis=1
            )
            df_display['Kg Acima da Meta']    = df_display['Kg_T3'].apply(lambda x: f"{x:,.0f}")
            df_display['Preço/kg Mês (R$)']   = df_display['Preco_Kg_Mes'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Valor Até Ano Anterior (R$)'] = df_display['Val_T1'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Valor Faixa Meta (R$)']       = df_display['Val_T2'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Valor Acima Meta (R$)']       = df_display['Val_T3'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão T1 (R$)'] = df_display['Com_T1'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão T2 (R$)'] = df_display['Com_T2'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão T3 (R$)'] = df_display['Com_T3'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão Total (R$)'] = df_display['Comissao_R$'].apply(lambda x: f"R$ {x:,.2f}")

            df_display = df_display[[
                'Distribuidor','Produto','Kg Ano Anterior','Meta Kg (mês)','Kg Mês','Δ Kg',
                'Kg Entre Ano Ant. e Meta','Kg Até Ano Anterior','Kg Acima da Meta',
                'Preço/kg Mês (R$)','Valor Até Ano Anterior (R$)',
                'Valor Faixa Meta (R$)','Valor Acima Meta (R$)',
                'Comissão T1 (R$)','Comissão T2 (R$)',
                'Comissão T3 (R$)','Comissão Total (R$)'
            ]]

        if df_display.empty:
            st.warning("❗ Nenhum dado encontrado para os filtros selecionados.")
        else:
            st.subheader(f"📅 Resultados – {selected_mes:02d}/{selected_ano}")
            st.dataframe(df_display, use_container_width=True)

            st.markdown("#### 📝 Legenda das Colunas (Mês Selecionado)")
            st.markdown("""
            - **Meta Kg (mês)**: soma das metas diárias (CSV ou Excel) para aquele distribuidor/sku no mês selecionado, convertida para inteiro.  
            - **Kg Ano Anterior**: soma de `total kg` no mesmo mês do ano anterior.  
            - **Kg Mês**: soma de `total kg` para o mês selecionado.  
            - **Δ Kg**: diferença entre `Kg Mês` e `Kg Ano Anterior`.  
            - **Kg Até Ano Anterior (Kg_T1)**: volume do mês que coincide com o volume até o ano anterior (min(total, prev)).  
            - **Kg Entre Ano Ant. e Meta**: `max(meta_kg – Total_Kg_Ant, 0)` (volume em que se aplica faixa T2).  
            - **Kg Acima da Meta (Kg_T3)**: volume acima da meta.  
            - **Preço/kg Mês (R$)**: `Faturamento_Mes / Kg Mês`.  
            - **Valor Até Ano Anterior (R$)**: `Kg_T1 * Preço/kg Mês`.  
            - **Valor Faixa Meta (R$)**: `Kg_T2 * Preço/kg Mês`.  
            - **Valor Acima Meta (R$)**: `Kg_T3 * Preço/kg Mês`.  
            - **Comissão T1 (R$)**: `Valor Até Ano Anterior * (pct1/100)`.  
            - **Comissão T2 (R$)**: `Valor Faixa Meta * (pct2/100)`.  
            - **Comissão T3 (R$)**: `Valor Acima Meta * (pct3/100)`.  
            - **Comissão Total (R$)**: soma de todas as faixas.
            """)

        # -------------------------------------------------------
        # Totais Consolidados (Mês Selecionado)
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Totais Consolidados (Mês Selecionado)**")
        totais_merge = (
            df_merge
            .groupby('nome_distribuidor', as_index=False)
            .agg(
                Total_Kg_Ant=('Total_Kg_Ant','sum'),
                Total_Kg_Mes=('Total_Kg_Mes','sum'),
                Sum_Meta_Kg=('meta_kg','sum'),
                Total_Fat_Mes=('Faturamento_Mes','sum'),
                Kg_T1_Total=('Kg_T1','sum'),
                Kg_T2_Total=('Kg_T2','sum'),
                Kg_T3_Total=('Kg_T3','sum'),
                Val_T1_Total=('Val_T1','sum'),
                Val_T2_Total=('Val_T2','sum'),
                Val_T3_Total=('Val_T3','sum'),
                Com_T1_Total=('Com_T1','sum'),
                Com_T2_Total=('Com_T2','sum'),
                Com_T3_Total=('Com_T3','sum'),
                Comissao_Total=('Comissao_R$','sum')
            )
            .rename(columns={'nome_distribuidor':'Distribuidor'})
        )
        totais_merge['Preco_Medio_Kg'] = totais_merge.apply(
            lambda r: (r['Total_Fat_Mes']/r['Total_Kg_Mes']) if r['Total_Kg_Mes']>0 else 0,
            axis=1
        )

        totais_merge['Kg Ano Anterior'] = totais_merge['Total_Kg_Ant'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Kg Mês']          = totais_merge['Total_Kg_Mes'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Meta Kg (mês)']   = totais_merge['Sum_Meta_Kg'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Preço Médio (R$/Kg)'] = totais_merge['Preco_Medio_Kg'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Kg Entre Ano Ant. e Meta'] = totais_merge.apply(
            lambda r: f"{max(r['Sum_Meta_Kg']-r['Total_Kg_Ant'],0):,.0f}", axis=1
        )
        totais_merge['Kg Até Ano Anterior']    = totais_merge['Kg_T1_Total'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Kg Acima da Meta']       = totais_merge['Kg_T3_Total'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Valor Até Ano Anterior (R$)'] = totais_merge['Val_T1_Total'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Valor Faixa Meta (R$)']       = totais_merge['Val_T2_Total'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Valor Acima Meta (R$)']       = totais_merge['Val_T3_Total'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Comissão T1 (R$)'] = totais_merge['Com_T1_Total'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Comissão T2 (R$)'] = totais_merge['Com_T2_Total'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Comissão T3 (R$)'] = totais_merge['Com_T3_Total'].apply(lambda x: f"R$ {x:,.2f}")
        totais_merge['Comissão Total (R$)'] = totais_merge['Comissao_Total'].apply(lambda x: f"R$ {x:,.2f}")

        totais_exib = totais_merge[[
            'Distribuidor',
            'Kg Ano Anterior','Kg Mês','Meta Kg (mês)','Kg Entre Ano Ant. e Meta',
            'Preço Médio (R$/Kg)','Kg Até Ano Anterior','Kg Acima da Meta',
            'Valor Até Ano Anterior (R$)','Valor Faixa Meta (R$)','Valor Acima Meta (R$)',
            'Comissão T1 (R$)','Comissão T2 (R$)','Comissão T3 (R$)','Comissão Total (R$)'
        ]]
        st.write(totais_exib)

        # ---- Cards de Totais Consolidados ----
        total_kg_ant_all  = totais_merge['Total_Kg_Ant'].sum()
        total_kg_mes_all  = totais_merge['Total_Kg_Mes'].sum()
        total_meta_kg_all = totais_merge['Sum_Meta_Kg'].sum()
        total_comissao_all = totais_merge['Comissao_Total'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Kg Ano Anterior (Total)", f"{total_kg_ant_all:,.0f}")
        col2.metric("Kg Mês (Total)", f"{total_kg_mes_all:,.0f}")
        col3.metric("Meta Kg (Total)", f"{total_meta_kg_all:,.0f}")
        col4.metric("Comissão Total (R$)", f"R$ {total_comissao_all:,.2f}")

        # ---- Definir cores para distribuidores ----
        color_sequence = [
            "#FF5733","#33FF57","#3357FF","#FF33A1","#A133FF",
            "#33FFF5","#FF8C33","#8CFF33","#338CFF","#FF338C",
            "#33A1FF","#A1FF33","#FF3333","#33FF33","#3333FF",
            "#FF33FF","#33FFFF","#FFFF33","#D35400","#27AE60",
            "#2980B9","#8E44AD","#16A085","#F39C12","#C0392B"
        ]
        n_dist = len(selected_dist)
        if n_dist > len(color_sequence):
            st.warning(f"Há mais distribuidores ({n_dist}) do que cores disponíveis ({len(color_sequence)}). Algumas cores poderão se repetir.")
        dist_colors = { dist: color_sequence[i % len(color_sequence)] for i, dist in enumerate(selected_dist) }

        # -------------------------------------------------------
        #  Gráfico de Comissões por Distribuidor (Mês Selecionado)
        # -------------------------------------------------------
        st.markdown("**Gráfico de Comissões por Distribuidor (Mês Selecionado)**")
        df_graf_mes = totais_merge[['Distribuidor','Comissao_Total']].copy()
        df_graf_mes['Comissao_Num'] = totais_merge['Comissao_Total'].replace(r'[R\$,]', '', regex=True).astype(float)

        fig_mes = px.bar(
            df_graf_mes,
            x='Distribuidor',
            y='Comissao_Num',
            text='Comissao_Num',
            color='Distribuidor',
            color_discrete_map=dist_colors,
            labels={'Comissao_Num':'Comissão (R$)'}
        )
        fig_mes.update_traces(
            texttemplate='R$ %{text:,.2f}',
            textposition='outside',
            marker_line_width=0.5
        )
        fig_mes.update_layout(
            uniformtext_minsize=12, uniformtext_mode='hide',
            yaxis_tickformat=",.2f",
            margin=dict(t=20,b=20,l=40,r=20),
            xaxis_title="Distribuidor", yaxis_title="Comissão (R$)",
            showlegend=False
        )
        st.plotly_chart(fig_mes, use_container_width=True)

        # -------------------------------------------------------
        #  Gráfico Anual de Comissões por Mês, com Projeção
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Gráfico Anual de Comissões por Mês e Distribuidor (com Projeções)**")

        # Primeiro: calcula as comissões efetivas para cada mês do ano selecionado
        df_annual = calcular_comissoes_mensais(
            df_fatur, df_meta_mensal,
            selected_dist, selected_produtos,
            pct1, pct2, pct3,
            selected_ano
        )
        if df_annual.empty:
            st.info("Não há dados de comissão anual para os filtros atuais.")
            return

        # Converte mês para string "MM"
        df_annual['mes_str'] = df_annual['mes'].apply(lambda x: f"{x:02d}")
        df_annual.rename(columns={'nome_distribuidor':'Distribuidor','Comissao_R$':'Comissao_Num'}, inplace=True)

        # Soma as comissões efetivas por mês (todas distribuidoras) para plotar as barras reais
        df_total_mes = df_annual.groupby('mes_str', as_index=False).agg(Total_Efetiva=('Comissao_Num','sum'))

        # ------------- PREPARAR PROJEÇÕES PARA MESES FUTUROS (DETALHADO COM CORES HACHURADAS) ----------------
        hoje = datetime.now()
        ano_atual = hoje.year
        mes_atual = hoje.month if selected_ano == ano_atual else 12

        # DataFrame para armazenar, para cada mês futuro, Com_Proj_T1, Com_Proj_T2 e Com_Proj_T3
        df_proj_detalhado = pd.DataFrame(columns=[
            'mes', 'mes_str', 'Com_Proj_T1', 'Com_Proj_T2', 'Com_Proj_T3'
        ])

        if selected_ano == ano_atual:
            # 1) Agrupa faturamento do ano anterior (selected_ano - 1) por distribuidor/produto/mês
            df_prev_full = df_fatur[
                (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                (df_fatur['ano'] == (selected_ano - 1))
            ].copy()
            if selected_produtos:
                df_prev_full = df_prev_full[df_prev_full['codigo_produto'].isin(selected_produtos)]
            df_prev_group = (
                df_prev_full
                .groupby(['nome_distribuidor', 'codigo_produto', 'mes'], as_index=False)
                .agg(
                    Total_Kg_Ant=('total kg', 'sum'),
                    Faturamento_Ant=('total df', 'sum')
                )
            )
            df_prev_group['Preco_Kg_Ant'] = df_prev_group.apply(
                lambda r: (r['Faturamento_Ant'] / r['Total_Kg_Ant']) if r['Total_Kg_Ant'] > 0 else 0,
                axis=1
            )

            # 2) Obtém metas mensais completas para o ano selecionado
            df_meta_full = df_meta_mensal.copy()  # contém ['nome_distribuidor','codigo_produto','ano','mes','meta_kg_mes']

            lista_proj_detalhada = []
            for mes in range(mes_atual + 1, 13):
                # Filtra faturamento anterior para este mês
                df_prev_m = df_prev_group[df_prev_group['mes'] == mes].copy()

                # Filtra meta deste mês para o ano corrente
                df_meta_m = df_meta_full[
                    (df_meta_full['ano'] == selected_ano) &
                    (df_meta_full['mes'] == mes)
                ][['nome_distribuidor', 'codigo_produto', 'meta_kg_mes']].copy()

                # Faz merge entre prev e meta
                df_merge_proj = pd.merge(
                    df_prev_m,
                    df_meta_m,
                    on=['nome_distribuidor', 'codigo_produto'],
                    how='left'
                ).fillna({
                    'Total_Kg_Ant': 0,
                    'Faturamento_Ant': 0,
                    'Preco_Kg_Ant': 0,
                    'meta_kg_mes': 0.0
                })

                # Função para calcular cada componente de comissão projetada (T1, T2 e T3)
                def calcular_comissoes_proj_detalhado(row):
                    prev = row['Total_Kg_Ant']
                    meta = row['meta_kg_mes']
                    preco = row['Preco_Kg_Ant']

                    # Assume que o faturamento futuro será igual à meta
                    if prev >= meta:
                        kg_t1 = meta
                        kg_t2 = 0
                        kg_t3 = 0
                    else:
                        kg_t1 = prev
                        kg_t2 = meta - prev
                        kg_t3 = 0

                    val_t1 = kg_t1 * preco
                    val_t2 = kg_t2 * preco
                    val_t3 = kg_t3 * preco  # será zero, mas mantido para estrutura

                    com_t1 = val_t1 * (pct1 / 100)
                    com_t2 = val_t2 * (pct2 / 100)
                    com_t3 = val_t3 * (pct3 / 100)

                    return pd.Series({
                        'Com_Proj_T1': com_t1,
                        'Com_Proj_T2': com_t2,
                        'Com_Proj_T3': com_t3
                    })

                if not df_merge_proj.empty:
                    comps = df_merge_proj.apply(calcular_comissoes_proj_detalhado, axis=1)
                    df_merge_proj[['Com_Proj_T1', 'Com_Proj_T2', 'Com_Proj_T3']] = comps

                    total_t1 = df_merge_proj['Com_Proj_T1'].sum()
                    total_t2 = df_merge_proj['Com_Proj_T2'].sum()
                    total_t3 = df_merge_proj['Com_Proj_T3'].sum()
                else:
                    total_t1 = total_t2 = total_t3 = 0.0

                lista_proj_detalhada.append({
                    'mes': mes,
                    'mes_str': f"{mes:02d}",
                    'Com_Proj_T1': total_t1,
                    'Com_Proj_T2': total_t2,
                    'Com_Proj_T3': total_t3
                })

            if lista_proj_detalhada:
                df_proj_detalhado = pd.DataFrame(lista_proj_detalhada)
            else:
                df_proj_detalhado = pd.DataFrame(columns=[
                    'mes', 'mes_str', 'Com_Proj_T1', 'Com_Proj_T2', 'Com_Proj_T3'
                ])

        # ---------- CONSTRUIR GRÁFICO COM BARRAS REAIS E PROJETADAS (COM CORES HACHURADAS DIFERENTES) -------------
        fig_annual_detalhado = go.Figure()

        # 1) Trace das comissões efetivas (barras sólidas)
        fig_annual_detalhado.add_trace(
            go.Bar(
                x=df_total_mes['mes_str'],
                y=df_total_mes['Total_Efetiva'],
                name='Comissão Efetiva',
                marker_color='steelblue',
                text=[f"R$ {v:,.2f}" for v in df_total_mes['Total_Efetiva']],
                textposition='outside'
            )
        )

        # 2) Traces das projeções detalhadas (barras empilhadas com hachuras/colorizações diferentes)
        if not df_proj_detalhado.empty:
            # T1 projetada (hachura padrão "/")
            fig_annual_detalhado.add_trace(
                go.Bar(
                    x=df_proj_detalhado['mes_str'],
                    y=df_proj_detalhado['Com_Proj_T1'],
                    name='Proj. T1',
                    marker_color='lightgray',
                    marker_pattern=dict(shape='/', size=6, solidity=0.5),
                    opacity=0.7,
                    text=[f"R$ {v:,.2f}" for v in df_proj_detalhado['Com_Proj_T1']],
                    textposition='inside'
                )
            )
            # T2 projetada (hachura "\" em cor levemente mais escura)
            fig_annual_detalhado.add_trace(
                go.Bar(
                    x=df_proj_detalhado['mes_str'],
                    y=df_proj_detalhado['Com_Proj_T2'],
                    name='Proj. T2',
                    marker_color='darkgray',
                    marker_pattern=dict(shape='x', size=6, solidity=0.1),
                    opacity=0.7,
                    text=[f"R$ {v:,.2f}" for v in df_proj_detalhado['Com_Proj_T2']],
                    textposition='inside'
                )
            )
            # T3 projetada (hachura "x" em cor ainda mais escura)
            fig_annual_detalhado.add_trace(
                go.Bar(
                    x=df_proj_detalhado['mes_str'],
                    y=df_proj_detalhado['Com_Proj_T3'],
                    name='Proj. T3',
                    marker_color='gray',
                    marker_pattern=dict(shape='\\', size=6, solidity=0.2),
                    opacity=0.7,
                    text=[f"R$ {v:,.2f}" for v in df_proj_detalhado['Com_Proj_T3']],
                    textposition='inside'
                )
            )

            # 3) Anotações com o valor total projetado (T1+T2+T3) acima de cada barra futura
            for idx, row in df_proj_detalhado.iterrows():
                total_mes = row['Com_Proj_T1'] + row['Com_Proj_T2'] + row['Com_Proj_T3']
                fig_annual_detalhado.add_annotation(
                    x=row['mes_str'],
                    y=total_mes * 1.02,
                    text=f"R$ {total_mes:,.2f}",
                    showarrow=False,
                    font=dict(size=11, color="black"),
                    bgcolor="rgba(255,255,255,0.7)"
                )

        # 4) Layout final (barras empilhadas em “stack”)
        fig_annual_detalhado.update_layout(
            barmode='stack',
            title_text=None,
            xaxis_title="Mês",
            yaxis_title="Comissão (R$)",
            yaxis_tickformat=",.2f",
            margin=dict(t=20, b=20, l=40, r=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig_annual_detalhado, use_container_width=True)

        # -------------------------------------------------------
        #  Cálculo e Gráfico da Média Trimestral de Comissões (barras)
        # -------------------------------------------------------
        # Objetivo: calcular a média móvel de 3 meses sobre a sequência completa de comissões
        # (real até o mês atual e projetado para meses futuros), levando em consideração
        # valores de faturamento a partir de novembro do ano anterior para compor a média de janeiro, etc.
        #
        # 1) Obter comissões mensais do ano anterior para meses 11 e 12
        # -------------------------------------------------------------
        df_prev_year = calcular_comissoes_mensais(
            df_fatur, df_meta_mensal,
            selected_dist, selected_produtos,
            pct1, pct2, pct3,
            selected_ano - 1
        )
        # Agrupa comissões por mês no ano anterior
        if not df_prev_year.empty:
            df_prev_year['mes_str'] = df_prev_year['mes'].apply(lambda x: f"{x:02d}")
            df_prev_year.rename(columns={'Comissao_R$': 'Comissao_Num'}, inplace=True)
            df_total_prev = (
                df_prev_year
                .groupby('mes_str', as_index=False)
                .agg(Com_Total=('Comissao_Num', 'sum'))
            )
        else:
            df_total_prev = pd.DataFrame(columns=['mes_str', 'Com_Total'])
        # Garante zeros para meses faltantes
        all_months = [f"{m:02d}" for m in range(1, 13)]
        df_total_prev = df_total_prev.set_index('mes_str').reindex(all_months, fill_value=0.0).reset_index()

        # Extrai valor de novembro e dezembro do ano anterior
        prev_nov = float(df_total_prev.loc[df_total_prev['mes_str'] == '11', 'Com_Total'].iloc[0])
        prev_dec = float(df_total_prev.loc[df_total_prev['mes_str'] == '12', 'Com_Total'].iloc[0])


        # 2) Construir série completa de comissões para o ano atual (mês a mês)
        # ---------------------------------------------------------------------
        # df_total_mes já contém Total_Efetiva para cada mês real do ano atual
        # Primeiro, criamos uma lista de 12 valores Com_Full, para cada mês "01" a "12":
        #   - Se mês ≤ mes_atual: Com_Full = Total_Efetiva (real)
        #   - Se mês > mes_atual:  Com_Full = soma(Com_Proj_T1 + Com_Proj_T2 + Com_Proj_T3)
        df_combinado = pd.DataFrame({'mes_str': all_months})
        df_combinado = df_combinado.merge(
            df_total_mes.rename(columns={'Total_Efetiva': 'Comissao_Real'}),
            on='mes_str', how='left'
        )
        df_combinado['Comissao_Real'] = df_combinado['Comissao_Real'].fillna(0.0)

        # Calcula comissão projetada total para meses futuros
        df_combinado['Comissao_Proj'] = 0.0
        if selected_ano == ano_atual and not df_proj_detalhado.empty:
            df_proj_total = df_proj_detalhado.copy()
            df_proj_total['Com_Total_Proj'] = (
                df_proj_total['Com_Proj_T1'] +
                df_proj_total['Com_Proj_T2'] +
                df_proj_total['Com_Proj_T3']
            )
            for _, row in df_proj_total.iterrows():
                mes_fut = row['mes_str']
                df_combinado.loc[
                    df_combinado['mes_str'] == mes_fut, 'Comissao_Proj'
                ] = row['Com_Total_Proj']

        # Cria lista Com_Full de 12 valores
        def get_com_full(r):
            m = int(r['mes_str'])
            if m <= mes_atual:
                return r['Comissao_Real']
            else:
                return r['Comissao_Proj']

        df_combinado['Comissao_Full'] = df_combinado.apply(get_com_full, axis=1)

        # 3) Construir lista estendida incluindo novembro e dezembro do ano anterior
        # ---------------------------------------------------------------------------
        # Índices: 0->nov_prev, 1->dec_prev, 2->jan, 3->fev, ..., 13->dez
        extended = [prev_nov, prev_dec] + df_combinado['Comissao_Full'].tolist()

        # 4) Calcular médias móveis de 3 meses ao longo de extended
        # --------------------------------------------------------
        medias = []
        for i in range(2, len(extended)):
            window = extended[i-2:i+1]
            medias.append(sum(window) / 3.0)
        # Agora medias é lista de comprimento 12, correspondendo a meses "01" a "12"

        # 5) Criar colunas para exibir no gráfico
        # ---------------------------------------
        df_combinado['Media_3M'] = medias  # média móvel para cada mês "01"-"12"
        df_combinado['Media_3M_Display'] = df_combinado.apply(
            lambda r: r['Media_3M'] if int(r['mes_str']) <= mes_atual else np.nan,
            axis=1
        )
        df_combinado['Media_3M_Proj'] = df_combinado.apply(
            lambda r: r['Media_3M'] if int(r['mes_str']) > mes_atual else np.nan,
            axis=1
        )

        # 6) Monta o gráfico de barras
        # -----------------------------
        fig_media = go.Figure()

        # 6.1) Barras verdes para Média 3M nos meses reais (≤ mes_atual)
        fig_media.add_trace(
            go.Bar(
                x=df_combinado['mes_str'],
                y=df_combinado['Media_3M_Display'],
                name='Média 3M (Real)',
                marker_color='rgba(0, 148, 185, 0.48)',
                text=[f"R$ {v:,.2f}" if not np.isnan(v) else "" for v in df_combinado['Media_3M_Display']],
                textposition='outside'
            )
        )

        # 6.2) Barras vermelhas hachuradas para Média 3M nos meses futuros (> mes_atual)
        fig_media.add_trace(
            go.Bar(
                x=df_combinado['mes_str'],
                y=df_combinado['Media_3M_Proj'],
                name='Média 3M (Projeção)',
                marker_color='red',
                marker_pattern=dict(shape='/', size=6, solidity=0.5),
                opacity=0.7,
                text=[f"R$ {v:,.2f}" if not np.isnan(v) else "" for v in df_combinado['Media_3M_Proj']],
                textposition='outside',
                textfont=dict(color='rgba(0, 0, 0, 0.79)')
            )
        )

        # 6.3) Layout final (sem alterações)
        fig_media.update_layout(
            barmode='overlay',
            title_text="Comissão Mensal e Média Trimestral (com Projeções)",
            xaxis_title="Mês",
            yaxis_title="Valor (R$)",
            yaxis_tickformat=",.2f",
            margin=dict(t=30, b=20, l=40, r=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig_media, use_container_width=True)





        # -------------------------------------------------------
        #  Tabelas de Valor por KG (por SKU por Distribuidor)
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Valor de Comissão por KG de cada SKU**")

        df_rate = df_merge[['nome_distribuidor','codigo_produto','Preco_Kg_Mes']].copy()
        df_rate['Valor_por_Kg_T1'] = df_rate['Preco_Kg_Mes'] * (pct1/100)
        df_rate['Valor_por_Kg_T2'] = df_rate['Preco_Kg_Mes'] * (pct2/100)
        df_rate['Valor_por_Kg_T3'] = df_rate['Preco_Kg_Mes'] * (pct3/100)

        for dist in selected_dist:
            st.markdown(f"**Distribuidor: {dist}**")
            df_dist = df_rate[df_rate['nome_distribuidor']==dist].copy()
            if df_dist.empty:
                st.write("Sem dados de SKU para este distribuidor.")
                continue

            col_t1, col_t2, col_t3 = st.columns(3)

            # Tabela T1
            with col_t1:
                st.subheader(f"T1 – Até ano anterior ({pct1:.3f}%)")
                df_t1 = df_dist[['codigo_produto','Valor_por_Kg_T1']].copy()
                df_t1.rename(columns={
                    'codigo_produto':'SKU',
                    'Valor_por_Kg_T1':f'R$/Kg T1 ({pct1:.3f}%)'
                }, inplace=True)
                df_t1[f'R$/Kg T1 ({pct1:.3f}%)'] = df_t1[f'R$/Kg T1 ({pct1:.3f}%)'].apply(
                    lambda x: f"R$ {x:,.3f}"
                )
                st.dataframe(df_t1.reset_index(drop=True), use_container_width=True)

            # Tabela T2
            with col_t2:
                st.subheader(f"T2 – Entre ano anterior e meta ({pct2:.3f}%)")
                df_t2 = df_dist[['codigo_produto','Valor_por_Kg_T2']].copy()
                df_t2.rename(columns={
                    'codigo_produto':'SKU',
                    'Valor_por_Kg_T2':f'R$/Kg T2 ({pct2:.3f}%)'
                }, inplace=True)
                df_t2[f'R$/Kg T2 ({pct2:.3f}%)'] = df_t2[f'R$/Kg T2 ({pct2:.3f}%)'].apply(
                    lambda x: f"R$ {x:,.3f}"
                )
                st.dataframe(df_t2.reset_index(drop=True), use_container_width=True)

            # Tabela T3
            with col_t3:
                st.subheader(f"T3 – Acima da meta ({pct3:.3f}%)")
                df_t3 = df_dist[['codigo_produto','Valor_por_Kg_T3']].copy()
                df_t3.rename(columns={
                    'codigo_produto':'SKU',
                    'Valor_por_Kg_T3':f'R$/Kg T3 ({pct3:.3f}%)'
                }, inplace=True)
                df_t3[f'R$/Kg T3 ({pct3:.3f}%)'] = df_t3[f'R$/Kg T3 ({pct3:.3f}%)'].apply(
                    lambda x: f"R$ {x:,.3f}"
                )
                st.dataframe(df_t3.reset_index(drop=True), use_container_width=True)

    else:
        st.sidebar.info("1) Carregue a base de faturamento. 2) Carregue o arquivo de metas (CSV ou Excel). 3) Configure filtros e clique em ‘Calcular’.")

if __name__ == "__main__":
    main()
