import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

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
    name = metas_file.name.lower()
    if name.endswith('.csv'):
        df_raw = pd.read_csv(metas_file, sep=';', dtype=str)
        sheets = {'sheet1': df_raw}
    else:
        sheets = pd.read_excel(metas_file, sheet_name=None, dtype=str)

    lista_metas = []
    for sheet_name, df_raw in sheets.items():
        df = df_raw.copy()
        df.columns = df.columns.str.strip()
        if 'NOME DISTRIBUIDOR' not in df.columns or 'COD PROD' not in df.columns:
            continue

        colunas = df.columns.tolist()
        idx_inicio_datas = colunas.index('COD PROD') + 1
        datas_cols = colunas[idx_inicio_datas:]

        melt = df.melt(
            id_vars=['NOME DISTRIBUIDOR', 'COD PROD'],
            value_vars=datas_cols,
            var_name='data_dia',
            value_name='meta_kg_dia'
        )
        melt['data_dia'] = pd.to_datetime(melt['data_dia'], dayfirst=True, errors='coerce')

        melt['meta_kg_dia'] = (
            melt['meta_kg_dia']
            .astype(str)
            .str.replace(',', '', regex=False)
        )
        melt['meta_kg_dia'] = pd.to_numeric(melt['meta_kg_dia'], errors='coerce').fillna(0.0)
        melt['meta_kg_dia'] = melt['meta_kg_dia'].round(2)

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
    df = df_metas.copy()
    df['ano'] = df['data_dia'].dt.year
    df['mes'] = df['data_dia'].dt.month
    df_mes = (
        df
        .groupby(['nome_distribuidor', 'codigo_produto', 'ano', 'mes'], as_index=False)
        .agg(meta_kg_mes=('meta_kg_dia', 'sum'))
    )
    df_mes['nome_distribuidor'] = df_mes['nome_distribuidor'].astype(str)
    df_mes['codigo_produto']    = df_mes['codigo_produto'].astype(str)
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
        # 1) Filtro faturamento para o mês e ano
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

        # 2) Faturamento ano anterior (mesmo mês) para Δ
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

        # 4) Cálculo de Δ
        df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']
        df_merge['Delta_R']  = df_merge['Faturamento_Mes'] - df_merge['Faturamento_Ant']

        # 5) Insere meta mensais
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

        # 6) Cálculo das comissões com penalização
        def calcular_coms(row):
            total = row['Total_Kg_Mes']
            prev  = row['Total_Kg_Ant']
            meta  = row['meta_kg']
            price = row['Preco_Kg_Mes']
            com1 = com2 = com3 = 0.0

            if total < prev:
                # Penalização por cada kg abaixo do ano anterior
                diff = prev - total
                com1 = - diff * price * (pct1 / 100)
            else:
                # total >= prev → sem penalização; calcula T2/T3
                if prev < meta:
                    vol_t2 = min(total, meta) - prev
                    if vol_t2 > 0:
                        com2 = vol_t2 * price * (pct2 / 100)
                    if total > meta:
                        vol_t3 = total - meta
                        com3 = vol_t3 * price * (pct3 / 100)
                else:
                    # caso prev ≥ meta, só T3 para o excedente sobre prev
                    if total > prev:
                        vol_t3 = total - prev
                        com3 = vol_t3 * price * (pct3 / 100)

            return pd.Series({'Com_T1': com1, 'Com_T2': com2, 'Com_T3': com3})

        if df_merge.empty:
            df_merge[['Com_T1', 'Com_T2', 'Com_T3']] = 0.0, 0.0, 0.0
        else:
            comps = df_merge.apply(calcular_coms, axis=1)
            df_merge[['Com_T1', 'Com_T2', 'Com_T3']] = comps[['Com_T1', 'Com_T2', 'Com_T3']]

        df_merge['Comissao_R$'] = df_merge['Com_T1'] + df_merge['Com_T2'] + df_merge['Com_T3']

        # 7) Agrupa a comissão total por distribuidor (número de passo ajustado)
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
    uploaded_fat = st.sidebar.file_uploader(
        "1) Carregue aqui o Excel da base de faturamento",
        type=["xlsx", "xls"], key="fat"
    )
    uploaded_meta = st.sidebar.file_uploader(
        "2) Carregue aqui o arquivo de metas diárias (CSV ou Excel)",
        type=["csv", "xlsx", "xls"], key="meta"
    )

    df_fatur = None
    df_metas_diarias = None
    df_meta_mensal  = None

    if uploaded_fat:
        try:
            df_raw = pd.read_excel(uploaded_fat)
            df_fatur = load_faturamento(df_raw)
        except Exception as e:
            st.sidebar.error(f"Falha ao ler Faturamento: {e}")
            df_fatur = None

    if uploaded_meta:
        try:
            df_metas_diarias = load_metas(uploaded_meta)
            df_meta_mensal  = aggregate_metas_mensais(df_metas_diarias)
        except Exception as e:
            st.sidebar.error(f"Falha ao ler Metas diárias: {e}")
            df_metas_diarias = None
            df_meta_mensal   = None

    distribuidores, anos, meses, produtos = [], [], [], []
    if df_fatur is not None:
        distribuidores = sorted(df_fatur['nome_distribuidor'].dropna().unique())
        anos = sorted(df_fatur['ano'].dropna().astype(int).unique())
        meses = list(range(1, 13))
        produtos = sorted(df_fatur['codigo_produto'].dropna().unique())

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
        pct1 = st.number_input("% Até volume do ano anterior (Penalização)", value=1.000, format="%.3f", step=0.001)
        pct2 = st.number_input("% Volume entre ano anterior e meta", value=1.000, format="%.3f", step=0.001)
        pct3 = st.number_input("% Acima da meta", value=2.000, format="%.3f", step=0.001)

        st.markdown("---")
        st.subheader("🎯 Comissão Extra por Cliente (T4)")
        if df_fatur is not None:
            clientes_unicos = sorted(df_fatur['codigo_cliente'].dropna().astype(str).unique())
        else:
            clientes_unicos = []
        clientes_selecionados = st.multiselect("Clientes (Código)", clientes_unicos, help="Clientes para aplicar comissão extra")
        pct4 = st.number_input("% Extra sobre total em KG (T4)", value=0.000, format="%.3f", step=0.001)


        st.markdown("---")

        btn_calcular = st.form_submit_button("🔍 Calcular")

    if btn_calcular:
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

        # -------------------------------------------------------
        # Cálculo Mês Selecionado
        # -------------------------------------------------------
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

            # 6) Cálculo das comissões em T1/T2/T3 com penalização
            def calcular_coms(row):
                total = row['Total_Kg_Mes']
                prev  = row['Total_Kg_Ant']
                meta  = row['meta_kg']
                price = row['Preco_Kg_Mes']
                com1 = com2 = com3 = 0.0

                if total < prev:
                    # Penalização por kg a menos que o ano anterior
                    diff = prev - total
                    com1 = - diff * price * (pct1 / 100)
                else:
                    # total >= prev: sem penalização, calcula T2/T3
                    if prev < meta:
                        vol_t2 = min(total, meta) - prev
                        if vol_t2 > 0:
                            com2 = vol_t2 * price * (pct2 / 100)
                        if total > meta:
                            vol_t3 = total - meta
                            com3 = vol_t3 * price * (pct3 / 100)
                    else:
                        if total > prev:
                            vol_t3 = total - prev
                            com3 = vol_t3 * price * (pct3 / 100)

                return pd.Series({'Com_T1': com1, 'Com_T2': com2, 'Com_T3': com3})

            if df_merge.empty:
                df_merge[['Com_T1','Com_T2','Com_T3']] = 0.0, 0.0, 0.0
            else:
                comps = df_merge.apply(calcular_coms, axis=1)
                df_merge[['Com_T1','Com_T2','Com_T3']] = comps[['Com_T1','Com_T2','Com_T3']]

            df_merge['Comissao_R$'] = df_merge['Com_T1'] + df_merge['Com_T2'] + df_merge['Com_T3']

            # -------------------------------------------------------
            # Cálculo da Comissão T4 – Extra por KG para clientes selecionados
            # -------------------------------------------------------
            if clientes_selecionados and pct4 > 0:
                df_fat_t4 = df_curr[df_curr['codigo_cliente'].astype(str).isin(clientes_selecionados)].copy()
                total_kg_t4 = df_fat_t4['total kg'].sum()
                total_df_t4 = df_fat_t4['total df'].sum()
                preco_medio_t4 = total_df_t4 / total_kg_t4 if total_kg_t4 > 0 else 0.0
                comissao_t4 = total_kg_t4 * preco_medio_t4 * (pct4 / 100)
            else:
                total_kg_t4 = 0
                preco_medio_t4 = 0
                comissao_t4 = 0



            df_display = df_merge.copy()
            df_display['Distribuidor'] = df_display['nome_distribuidor']
            df_display['Produto']     = df_display['codigo_produto']
            df_display['Kg Ano Anterior'] = df_display['Total_Kg_Ant'].apply(lambda x: f"{x:,.0f}")
            df_display['Meta Kg (mês)']   = df_display['meta_kg'].apply(lambda x: f"{x:,.0f}")
            df_display['Kg Mês']          = df_display['Total_Kg_Mes'].apply(lambda x: f"{x:,.0f}")
            df_display['Δ Kg']            = df_display['Delta_Kg'].apply(lambda x: f"{x:,.0f}")
            # Exibir apenas colunas essenciais agora que Kg_T1/T2/T3 e Val_T1/T2/T3 não existem:
            df_display['Penalização (R$)']   = df_display['Com_T1'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão T2 (R$)']   = df_display['Com_T2'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão T3 (R$)']   = df_display['Com_T3'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Comissão Total (R$)'] = df_display['Comissao_R$'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['Preço/kg Mês (R$)']  = df_display['Preco_Kg_Mes'].apply(lambda x: f"R$ {x:,.2f}")
            # Caso queira exibir “Kg até Ano Anterior” (apenas para referência, sem impactar cálculo):
            df_display['Volume Até Ano Anterior (Kg)'] = df_display.apply(
                lambda r: f"{min(r['Total_Kg_Mes'], r['Total_Kg_Ant']):,.0f}", axis=1
            )


            df_display = df_display[
                [
                    'Distribuidor',
                    'Produto',
                    'Kg Ano Anterior',
                    'Meta Kg (mês)',
                    'Kg Mês',
                    'Δ Kg',
                    # Remova ou comente a linha abaixo se não criou essa coluna:
                    'Volume Até Ano Anterior (Kg)',
                    'Preço/kg Mês (R$)',
                    'Penalização (R$)',
                    'Comissão T2 (R$)',
                    'Comissão T3 (R$)',
                    'Comissão Total (R$)'
                ]
            ]



        if df_display.empty:
            st.warning("❗ Nenhum dado encontrado para os filtros selecionados.")
        else:
            st.subheader(f"📅 Resultados – {selected_mes:02d}/{selected_ano}")
            st.dataframe(df_display, use_container_width=True)

            st.markdown("#### 📝 Legenda das Colunas (Mês Selecionado)")
            st.markdown("""
            - **Meta Kg (mês)**: soma de todas as metas diárias para o distribuidor/sku no mês selecionado, arredondada para inteiro.  
            - **Kg Ano Anterior**: total de quilos vendidos no mesmo mês do ano anterior.  
            - **Kg Mês**: total de quilos vendidos no mês selecionado deste ano.  
            - **Δ Kg**: diferença entre `Kg Faturados este Mês` e `Kg Mês/Ano Anterior`.  
            - **Volume Até Ano Anterior (Kg)**: Mostra quantos quilos deste mês equivalem ao ano anterior.  
            - **Penalização (R$)**:  
            - Se `Kg Mês < Kg Ano Anterior`, esse valor é calculado como  
                `(Kg Ano Anterior – Kg Mês) × Preço/kg Mês × (% Penalização)`, sempre aparece como número **negativo**.  
            - Caso contrário, vale **R$ 0,00**.  
            - **Comissão T2 (R$)**:  
            - Se `Kg Mês > Kg Ano Anterior` **e** `Kg Ano Anterior < Meta Kg`, então  
                `Volume na Faixa T2 = (Meta Kg - Kg Mês) - Kg Ano Anterior`,  
                e a comissão é `(Volume na Faixa T2) × Preço/kg Mês × (% T2)`.  
            - Em todos os outros casos, vale **R$ 0,00**.  
            - **Comissão T3 (R$)**:  
            - Se `Kg Mês > Meta Kg`, então `Volume na Faixa T3 = Kg Mês – Meta Kg`,  
                e a comissão é `(Volume na Faixa T3) × Preço/kg Mês × (% T3)`.  
            - Em todos os outros casos, vale **R$ 0,00**.  
            - Comissão Total (R$):  Penalização + Comissão T2 + Comissão T3.  
            """)


        # -------------------------------------------------------
        # Totais Consolidados (Mês Selecionado)
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Totais Consolidados (Mês Selecionado)**")

        # 1) Agrupa totais, somando apenas as colunas que existem (Com_T1, Com_T2, Com_T3)
        totais_merge = (
            df_merge
            .groupby('nome_distribuidor', as_index=False)
            .agg(
                Total_Kg_Ant=('Total_Kg_Ant', 'sum'),
                Total_Kg_Mes=('Total_Kg_Mes', 'sum'),
                Sum_Meta_Kg=('meta_kg', 'sum'),
                Total_Fat_Mes=('Faturamento_Mes', 'sum'),
                # Somando apenas as colunas de comissão (T1/T2/T3) que existem
                Com_T1_Total=('Com_T1', 'sum'),
                Com_T2_Total=('Com_T2', 'sum'),
                Com_T3_Total=('Com_T3', 'sum'),
                Comissao_Total=('Comissao_R$', 'sum')
            )
            .rename(columns={'nome_distribuidor': 'Distribuidor'})
        )

        # 2) Calcular Δ Kg total por distribuidor
        totais_merge['Delta_Kg_Total'] = totais_merge['Total_Kg_Mes'] - totais_merge['Total_Kg_Ant']

        # 3) Preço médio por kg
        totais_merge['Preco_Medio_Kg'] = totais_merge.apply(
            lambda r: (r['Total_Fat_Mes'] / r['Total_Kg_Mes']) if r['Total_Kg_Mes'] > 0 else 0,
            axis=1
        )

        # 4) Formatação das colunas numéricas para exibição
        totais_merge['Kg Ano Anterior'] = totais_merge['Total_Kg_Ant'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Kg Mês']          = totais_merge['Total_Kg_Mes'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Δ Kg']            = totais_merge['Delta_Kg_Total'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Meta Kg (mês)']   = totais_merge['Sum_Meta_Kg'].apply(lambda x: f"{x:,.0f}")
        totais_merge['Preço Médio (R$/Kg)'] = totais_merge['Preco_Medio_Kg'].apply(lambda x: f"R$ {x:,.2f}")
        # Penalização agregada (T1)
        totais_merge['Penalização (R$)'] = totais_merge['Com_T1_Total'].apply(lambda x: f"R$ {x:,.2f}")
        # Comissão da faixa T2 agregada
        totais_merge['Comissão T2 (R$)'] = totais_merge['Com_T2_Total'].apply(lambda x: f"R$ {x:,.2f}")
        # Comissão da faixa T3 agregada
        totais_merge['Comissão T3 (R$)'] = totais_merge['Com_T3_Total'].apply(lambda x: f"R$ {x:,.2f}")
        # Comissão total agregada
        totais_merge['Comissão Total (R$)'] = totais_merge['Comissao_Total'].apply(lambda x: f"R$ {x:,.2f}")

        # 5) Selecionar colunas para exibir em tabela (somente as que existem)
        totais_exib = totais_merge[
            [
                'Distribuidor',
                'Kg Ano Anterior',
                'Kg Mês',
                'Δ Kg',
                'Meta Kg (mês)',
                'Preço Médio (R$/Kg)',
                'Penalização (R$)',
                'Comissão T2 (R$)',
                'Comissão T3 (R$)',
                'Comissão Total (R$)'
            ]
        ]
        st.write(totais_exib)
        st.markdown("---")

        # 6) Cálculo dos totais gerais (soma de todas as linhas, para os cartões de métrica)
        total_kg_ant_all   = totais_merge['Total_Kg_Ant'].sum()
        total_kg_mes_all   = totais_merge['Total_Kg_Mes'].sum()
        total_delta_kg_all = totais_merge['Delta_Kg_Total'].sum()
        total_meta_kg_all  = totais_merge['Sum_Meta_Kg'].sum()
        total_comissao_all = totais_merge['Comissao_Total'].sum()

        # 7) Exibir métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Kg Ano Anterior (Total)", f"{total_kg_ant_all:,.0f}")
        col2.metric("Kg Mês (Total)", f"{total_kg_mes_all:,.0f}")
        col3.metric("Δ Kg (Total)", f"{total_delta_kg_all:,.0f}")
        col4.metric("Meta Kg (Total)", f"{total_meta_kg_all:,.0f}")
        col5.metric("Comissão Total (R$)", f"R$ {total_comissao_all:,.2f}")

        if comissao_t4 > 0:
            st.markdown("---")
            st.markdown("**💰 Comissão T4 – Extra por Clientes Selecionados**")
            col_t4_1, col_t4_2, col_t4_3 = st.columns(3)
            col_t4_1.metric("Total KG (Clientes)", f"{total_kg_t4:,.0f}")
            col_t4_2.metric("Preço Médio (R$/Kg)", f"R$ {preco_medio_t4:,.2f}")
            col_t4_3.metric("Comissão T4 (R$)", f"R$ {comissao_t4:,.2f}")

            # Total geral com T4
            st.markdown("---")
            st.markdown("### 💵 Comissão Total + T4")
            st.metric("Comissão Total com T4 (R$)", f"R$ {total_comissao_all + comissao_t4:,.2f}")


        # -------------------------------------------------------
        #  Definindo as cores para cada distribuidor ANTES de usar no gráfico
        # -------------------------------------------------------
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
        # Aqui criamos o mapeamento de cada distribuidor para uma cor
        dist_colors = { dist: color_sequence[i % len(color_sequence)] for i, dist in enumerate(selected_dist) }

        # 8) (Opcional) Gráfico de Δ Kg por distribuidor
        fig_delta = px.bar(
            totais_merge,
            x='Distribuidor',
            y='Delta_Kg_Total',
            text='Delta_Kg_Total',
            color='Distribuidor',
            color_discrete_map=dist_colors,  # AGORA `dist_colors` já existe
            labels={'Delta_Kg_Total':'Δ Kg'}
        )
        fig_delta.update_traces(
            texttemplate='%{text:,.0f}',
            textposition='outside',
            marker_line_width=0.5
        )
        fig_delta.update_layout(
            yaxis_tickformat=",.0f",
            margin=dict(t=20,b=20,l=40,r=20),
            xaxis_title="Distribuidor", yaxis_title="Δ Kg",
            showlegend=False
        )

        # Cores para distribuidores
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
        #  Gráfico de Comissões por Distribuidor (Mês Selecionado) – HACHURA COM FUNDO BRANCO E RÓTULOS COM FUNDO
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Gráfico de Comissões por Distribuidor (Mês Selecionado)**")

        # 1) Extrai “base” (T1+T2+T3) e renomeia
        df_base = totais_merge[['Distribuidor', 'Comissao_Total']].copy()
        df_base.rename(columns={'Comissao_Total': 'comissao_base'}, inplace=True)

        # 2) Calcula T4 por distribuidor
        if clientes_selecionados and pct4 > 0:
            df_t4 = df_curr[
                df_curr['codigo_cliente'].astype(str).isin(clientes_selecionados)
            ].copy()

            df_t4_dist = (
                df_t4
                .groupby('nome_distribuidor', as_index=False)
                .agg(total_kg_t4=('total kg', 'sum'),
                     total_df_t4=('total df', 'sum'))
            )
            df_t4_dist['preco_medio_t4'] = df_t4_dist.apply(
                lambda r: (r['total_df_t4'] / r['total_kg_t4']) if r['total_kg_t4'] > 0 else 0.0,
                axis=1
            )
            df_t4_dist['comissao_t4'] = (
                df_t4_dist['total_kg_t4'] * df_t4_dist['preco_medio_t4'] * (pct4 / 100)
            )
            df_t4_dist.rename(columns={'nome_distribuidor': 'Distribuidor'}, inplace=True)
        else:
            df_t4_dist = pd.DataFrame({
                'Distribuidor': selected_dist,
                'comissao_t4': [0.0] * len(selected_dist)
            })

        # 3) Mescla “base” e T4
        df_plot = pd.merge(
            df_base,
            df_t4_dist[['Distribuidor', 'comissao_t4']],
            on='Distribuidor',
            how='left'
        ).fillna({'comissao_t4': 0.0})

        # 4) Constrói barras empilhadas
        fig_mes = go.Figure()

        for dist in selected_dist:
            color = dist_colors.get(dist, "#CCCCCC")
            val_base = float(df_plot.loc[df_plot['Distribuidor'] == dist, 'comissao_base'].iloc[0])
            val_t4   = float(df_plot.loc[df_plot['Distribuidor'] == dist, 'comissao_t4'].iloc[0])
            total    = val_base + val_t4

            # 4.1) Traço “base” (sempre negativo ou zero), sem texto embutido — usaremos anotação com fundo
            fig_mes.add_trace(
                go.Bar(
                    x=[dist],
                    y=[val_base],
                    name='Comissão Base',
                    marker_color=color,
                    hovertemplate=f"Base (T1+T2+T3): R$ {val_base:,.2f}<extra></extra>",
                    showlegend=False
                )
            )

            # 4.2) Traço “T4” (sempre positivo), posicionado sobre a “base”
            if val_t4 > 0:
                # Se, mesmo somando T4, total for negativo => padrão '/'
                # Caso contrário (total ≥ 0), use '\'
                pattern_shape = "/" if total < 0 else "\\"
                fig_mes.add_trace(
                    go.Bar(
                        x=[dist],
                        y=[val_t4],
                        name='Comissão T4',
                        marker_color=color,
                        marker_pattern=dict(
                            shape=pattern_shape,
                            fgcolor=color,
                            bgcolor="white",  # fundo branco em vez de transparente
                            size=8
                        ),
                        hovertemplate=f"T4 (Extra): R$ {val_t4:,.2f}<extra></extra>",
                        showlegend=False
                    )
                )

            # 4.3) Anotação “Base” (valor absoluto) dentro do segmento negativo
            if val_base < 0:
                # Posicionamos no meio da barra “base”, em y = val_base/2
                fig_mes.add_annotation(
                    x=dist,
                    y=val_base / 2,
                    text=f"R$ {val_base:,.2f}",
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="black",     # cor do texto
                        family="Arial"
                    ),
                    bgcolor="white",       # fundo branco para destacar
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2
                )

            # 4.4) Anotação “T4” (valor) dentro do segmento hachurado
            if val_t4 > 0:
                # Posicionamos em y = val_base + val_t4/2
                fig_mes.add_annotation(
                    x=dist,
                    y=val_base + (val_t4 / 2),
                    text=f"R$ {val_t4:,.2f}",
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="black",
                        family="Arial"
                    ),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2
                )

            # 4.5) Anotação “Total” (valor final) — se total < 0, fica dentro; se total ≥ 0, acima de zero
            if total < 0:
                # Dentro da área negativa, acima do segmento T4 (ou base, se T4=0)
                base_y = val_base + (val_t4 if val_t4 > 0 else 0)
                fig_mes.add_annotation(
                    x=dist,
                    y=base_y + (abs(total - base_y) / 2),
                    text=f"Total: R$ {total:,.2f}",
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="white",
                        family="Arial",
                        weight="bold"
                    ),
                    bgcolor="black",     # fundo escuro para contraste na área negativa
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=2
                )
            else:
                # Acima de zero, dentro da área positiva
                fig_mes.add_annotation(
                    x=dist,
                    y=total / 2,
                    text=f"Total: R$ {total:,.2f}",
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="black",
                        family="Arial",
                        weight="bold"
                    ),
                    bgcolor="lightgrey", # fundo claro para destaque
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2
                )

        # 5) Layout geral
        fig_mes.update_layout(
            barmode='stack',
            title_text="Comissão por Distribuidor (Detalhado: Base + T4)",
            xaxis_title="",
            yaxis_title="Comissão (R$)",
            yaxis_tickformat=",.2f",
            margin=dict(t=30, b=60, l=40, r=20)
        )

        # 6) Remove rótulos padrão do eixo x
        fig_mes.update_xaxes(
            ticktext=[''] * len(selected_dist),
            tickvals=selected_dist
        )

        # 7) Adiciona anotações de rótulo de categoria com fundo cinza
        for dist in selected_dist:
            fig_mes.add_annotation(
                x=dist,
                y=0,
                text=dist,
                showarrow=False,
                yshift=-20,
                xanchor="center",
                font=dict(
                    size=12,
                    color="black"
                ),
                bgcolor="lightgrey"
            )

        # 8) Exibe no Streamlit com chave única
        st.plotly_chart(
            fig_mes,
            use_container_width=True,
            key="grafico_comissao_mes"
        )




        df_annual = calcular_comissoes_mensais(
            df_fatur, df_meta_mensal,
            selected_dist, selected_produtos,
            pct1, pct2, pct3,
            selected_ano
        )
        if df_annual.empty:
            st.info("Não há dados de comissão anual para os filtros atuais.")
            return

        # -------------------------------------------------------
        # 1) Comissões T1+T2+T3 por Distribuidor → df_annual
        # -------------------------------------------------------
        df_annual['mes_str'] = df_annual['mes'].apply(lambda x: f"{x:02d}")
        df_annual.rename(
            columns={'nome_distribuidor': 'Distribuidor', 'Comissao_R$': 'Comissao_Num'},
            inplace=True
        )

        # -------------------------------------------------------
        # 2) Agrupa para ter o total “real” de comissão (T1+T2+T3) por mês
        # -------------------------------------------------------
        df_total_mes = (
            df_annual
            .groupby('mes_str', as_index=False)
            .agg(Total_Efetiva=('Comissao_Num', 'sum'))
        )

        # -------------------------------------------------------
        # 3) Descobre até qual mês há dados reais (ano atual vs. selecionado)
        # -------------------------------------------------------
        hoje = datetime.now()
        ano_atual = hoje.year
        mes_atual = hoje.month if selected_ano == ano_atual else 12

        # -------------------------------------------------------
        # 4) Calcula T4 mensal (para cada mês de 1 até mes_atual)
        # -------------------------------------------------------
        # Monta lista de dicionários: {'mes_str': '01', 'Comissao_T4': valor}, ...
        t4_por_mes = []
        if clientes_selecionados and pct4 > 0:
            for m in range(1, mes_atual + 1):
                # Filtra faturamento do mês m apenas para distribuidores selecionados e clientes T4
                df_fat_t4_m = df_fatur[
                    (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                    (df_fatur['ano'] == selected_ano) &
                    (df_fatur['mes'] == m) &
                    (df_fatur['codigo_cliente'].astype(str).isin(clientes_selecionados))
                ].copy()

                total_kg_m = df_fat_t4_m['total kg'].sum()
                total_df_m = df_fat_t4_m['total df'].sum()
                preco_medio_m = (total_df_m / total_kg_m) if total_kg_m > 0 else 0.0
                com_t4_m = total_kg_m * preco_medio_m * (pct4 / 100)

                t4_por_mes.append({'mes_str': f"{m:02d}", 'Comissao_T4': com_t4_m})
        else:
            # Se não houver clientes ou pct4=0, comissao T4 = 0 para todos os meses reais
            for m in range(1, mes_atual + 1):
                t4_por_mes.append({'mes_str': f"{m:02d}", 'Comissao_T4': 0.0})

        df_t4_mes = pd.DataFrame(t4_por_mes)

        # -------------------------------------------------------
        # 5) Junta df_total_mes (T1+T2+T3) com df_t4_mes e soma T4 → atualiza Total_Efetiva
        # -------------------------------------------------------
        df_total_mes = pd.merge(
            df_total_mes,
            df_t4_mes,
            on='mes_str',
            how='right'  # garante que cada mes_str real tenha uma linha
        ).fillna({'Total_Efetiva': 0.0, 'Comissao_T4': 0.0})

        df_total_mes['Total_Efetiva'] = df_total_mes['Total_Efetiva'] + df_total_mes['Comissao_T4']

        # -------------------------------------------------------
        # 6) Preenche todos os meses (01–12), colocando 0.0 para meses sem dados
        # -------------------------------------------------------
        all_months = [f"{m:02d}" for m in range(1, 13)]
        df_total_mes = (
            df_total_mes
            .set_index('mes_str')
            .reindex(all_months, fill_value=0.0)
            .reset_index()
        )

        # -------------------------------------------------------
        # 7) Projeções detalhadas (meses futuros) — sem alteração
        # -------------------------------------------------------
        df_proj_detalhado = pd.DataFrame(columns=[
            'mes', 'mes_str', 'Com_Proj_T1', 'Com_Proj_T2', 'Com_Proj_T3'
        ])

        if selected_ano == ano_atual:
            df_prev_full = df_fatur[
                (df_fatur['nome_distribuidor'].isin(selected_dist)) &
                (df_fatur['ano'] == (selected_ano - 1))
            ].copy()
            if selected_produtos:
                df_prev_full = df_prev_full[
                    df_prev_full['codigo_produto'].isin(selected_produtos)
                ]
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

            df_meta_full = df_meta_mensal.copy()
            lista_proj_detalhada = []
            for mes in range(mes_atual + 1, 13):
                df_prev_m = df_prev_group[df_prev_group['mes'] == mes].copy()
                df_meta_m = df_meta_full[
                    (df_meta_full['ano'] == selected_ano) &
                    (df_meta_full['mes'] == mes)
                ][['nome_distribuidor', 'codigo_produto', 'meta_kg_mes']].copy()

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

                def calcular_comissoes_proj_detalhado(row):
                    prev = row['Total_Kg_Ant']
                    meta = row['meta_kg_mes']
                    preco = row['Preco_Kg_Ant']
                    com1 = com2 = com3 = 0.0

                    if meta < prev:
                        diff = prev - meta
                        com1 = - diff * preco * (pct1 / 100)
                    elif meta == prev:
                        com1 = com2 = com3 = 0.0
                    else:
                        vol_t2 = meta - prev
                        com2 = vol_t2 * preco * (pct2 / 100)
                        com3 = 0.0

                    return pd.Series({
                        'Com_Proj_T1': com1,
                        'Com_Proj_T2': com2,
                        'Com_Proj_T3': com3
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

        # -------------------------------------------------------
        # 8) Comissões do ano anterior (para projeção de novos meses) — sem alteração
        # -------------------------------------------------------
        df_prev_year = calcular_comissoes_mensais(
            df_fatur, df_meta_mensal,
            selected_dist, selected_produtos,
            pct1, pct2, pct3,
            selected_ano - 1
        )
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

        all_months = [f"{m:02d}" for m in range(1, 13)]

        # Se não existirem meses na base anterior, preenche com zeros
        df_total_prev = (
            df_total_prev
            .set_index('mes_str')
            .reindex(all_months, fill_value=0.0)
            .reset_index()
        )

        # -------------------------------------------------------
        # 9) Monta DataFrame combinado (real até mes_atual + projeções)
        # -------------------------------------------------------
        df_combinado = pd.DataFrame({'mes_str': all_months})
        df_combinado = df_combinado.merge(
            df_total_mes.rename(columns={'Total_Efetiva': 'Comissao_Real'}),
            on='mes_str',
            how='left'
        )
        df_combinado['Comissao_Real'] = df_combinado['Comissao_Real'].fillna(0.0)

        # Preenche coluna de projeção com 0.0 por enquanto
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

        # Função auxiliar para escolher entre real e projeção
        def get_com_full(r):
            m = int(r['mes_str'])
            if m <= mes_atual:
                return r['Comissao_Real']
            else:
                return r['Comissao_Proj']

        df_combinado['Comissao_Full'] = df_combinado.apply(get_com_full, axis=1)

        # -------------------------------------------------------
        # 10) Extende lista com nov/dez do ano anterior e meses do ano atual
        # -------------------------------------------------------
        prev_nov = float(df_total_prev.loc[df_total_prev['mes_str'] == '11', 'Com_Total'].iloc[0])
        prev_dec = float(df_total_prev.loc[df_total_prev['mes_str'] == '12', 'Com_Total'].iloc[0])
        extended = [prev_nov, prev_dec] + df_combinado['Comissao_Full'].tolist()
        extended_month_labels = (
            [f"11_{selected_ano-1}", f"12_{selected_ano-1}"]
            + [f"{m}_{selected_ano}" for m in all_months]
        )

        # -------------------------------------------------------
        # 11) Cálculo de média móvel 3 meses e hover_info
        # -------------------------------------------------------
        medias = []
        hover_infos = []
        for i in range(2, len(extended)):
            window_vals = extended[i-2:i+1]
            window_labels = extended_month_labels[i-2:i+1]
            medias.append(sum(window_vals) / 3.0)

            info_lines = []
            for lbl, val in zip(window_labels, window_vals):
                info_lines.append(f"Mês {lbl}: R$ {val:,.2f}")
            hover_infos.append("<br>".join(info_lines))

        df_combinado['Media_3M'] = medias
        df_combinado['hover_info'] = hover_infos
        df_combinado['Media_3M_Display'] = df_combinado.apply(
            lambda r: r['Media_3M'] if int(r['mes_str']) <= mes_atual else np.nan,
            axis=1
        )
        df_combinado['Media_3M_Proj'] = df_combinado.apply(
            lambda r: r['Media_3M'] if int(r['mes_str']) > mes_atual else np.nan,
            axis=1
        )

        # -------------------------------------------------------
        # 12) Cálculo da média anual (horizontal)
        # -------------------------------------------------------
        media_anual = df_combinado['Comissao_Full'].sum() / 12.0


        # Exibe um card com o valor da média anual
        st.markdown("---")
        st.markdown("### 📊 Média Anual de Comissão")
        st.metric(
            label="Média Anual (R$/mês)",
            value=f"R$ {media_anual:,.2f}"
)

        fig_media = go.Figure()

        # Barras reais de Média 3M
        fig_media.add_trace(
            go.Bar(
                x=df_combinado['mes_str'],
                y=df_combinado['Media_3M_Display'],
                name='Média 3M (Real)',
                marker_color='rgba(0, 148, 185, 0.48)',
                customdata=df_combinado['hover_info'],
                hovertemplate='%{customdata}<extra></extra>',
                text=[f"R$ {v:,.2f}" if not np.isnan(v) else "" for v in df_combinado['Media_3M_Display']],
                textposition='outside'
            )
        )

        # Barras projetadas de Média 3M
        fig_media.add_trace(
            go.Bar(
                x=df_combinado['mes_str'],
                y=df_combinado['Media_3M_Proj'],
                name='Média 3M (Projeção)',
                marker_color='red',
                marker_pattern=dict(shape='/', size=6, solidity=0.5),
                opacity=0.7,
                customdata=df_combinado['hover_info'],
                hovertemplate='%{customdata}<extra></extra>',
                text=[f"R$ {v:,.2f}" if not np.isnan(v) else "" for v in df_combinado['Media_3M_Proj']],
                textposition='outside',
                textfont=dict(color='rgba(0, 0, 0, 0.79)')
            )
        )

        # Linha da média anual
        fig_media.add_trace(
            go.Scatter(
                x=all_months,
                y=[media_anual] * 12,
                mode='lines',
                name='Média Anual',
                line=dict(color='black', dash='dash'),
                hovertemplate=f'Média Anual: R$ {media_anual:,.2f}<extra></extra>'
            )
        )

        # ------------------ ANOTAÇÃO DO VALOR DA MÉDIA ------------------
        # Posiciona o texto no último mês (all_months[-1]) exatamente em y=media_anual
        fig_media.add_annotation(
            x=all_months[-1],               # mês mais à direita, ex: "12"
            y=media_anual,                  # valor exato da média
            text=f"R$ {media_anual:,.2f}",  # formata como R$ 12.345,67
            showarrow=True,                 # desenha uma flechinha apontando para o ponto
            arrowhead=2,                    # estilo da flecha
            ax=0,                           # deslocamento horizontal do texto (0 → alinhado em x)
            ay=-20,                         # deslocamento vertical (ajusta para ficar acima da linha)
            font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)",# fundo semitransparente para destacar o texto
            bordercolor="black",
            borderwidth=1
        )
        # ---------------------------------------------------------------


        fig_media.update_layout(
            barmode='overlay',
            title_text="Comissão Mensal e Média Trimestral (com Projeções e Linha Anual)",
            xaxis_title="Mês",
            yaxis_title="Valor (R$)",
            yaxis_tickformat=",.2f",
            margin=dict(t=30, b=20, l=40, r=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(
            fig_media,
            use_container_width=True,
            key="media_anual_proj"
        )

        # -------------------------------------------------------
        #  Tabelas de Valor por KG (por SKU por Distribuidor)
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Valor de Comissão por KG de cada SKU**")

        df_rate = df_merge[['nome_distribuidor','codigo_produto','Preco_Kg_Mes']].copy()
        df_rate['Penal_por_Kg'] = df_rate['Preco_Kg_Mes'] * (pct1/100)
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
                st.subheader(f"Penalização por Kg (-{pct1:.3f}%)")
                df_t1 = df_dist[['codigo_produto','Penal_por_Kg']].copy()
                df_t1.rename(columns={
                  'codigo_produto':'SKU',
                   'Penal_por_Kg':f'R$/Kg Penalização -({pct1:.3f}%)'
                }, inplace=True)
                df_t1[f'R$/Kg Penalização -({pct1:.3f}%)'] = df_t1[f'R$/Kg Penalização -({pct1:.3f}%)'].apply(
                   lambda x: f"- R$ {x:,.2f}"
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
                    lambda x: f"R$ {x:,.2f}"
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
                    lambda x: f"R$ {x:,.2f}"
                )
                st.dataframe(df_t3.reset_index(drop=True), use_container_width=True)
            
        # -------------------------------------------------------
        # --- Nova seção: Δ Kg por SKU em small multiples (Fora do loop de distribuidores)
        # -------------------------------------------------------
        def montar_delta_por_mes(df_fatur, df_meta_mensal, selected_dist, selected_produtos, selected_ano):
            """
            Retorna um DataFrame com as colunas:
            - mes (1 a 12)
            - nome_distribuidor
            - codigo_produto
            - Delta_Kg (Total_Kg_Mes - Total_Kg_Ant)
            """
            lista = []
            for mes in range(1, 13):
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
                    .agg(Total_Kg_Mes=('total kg', 'sum'))
                )

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
                    .agg(Total_Kg_Ant=('total kg', 'sum'))
                )

                df_merge = pd.merge(
                    df_current, df_prev_group,
                    on=['nome_distribuidor', 'codigo_produto'],
                    how='left'
                ).fillna({'Total_Kg_Ant': 0})
                df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']

                df_merge = df_merge[['nome_distribuidor', 'codigo_produto', 'Delta_Kg']].copy()
                df_merge['mes'] = mes
                lista.append(df_merge)

            if lista:
                return pd.concat(lista, ignore_index=True)
            else:
                return pd.DataFrame(columns=['mes', 'nome_distribuidor', 'codigo_produto', 'Delta_Kg'])


        # 2) Vamos calcular e desenhar os “small multiples” UMA ÚNICA VEZ, após o loop de distribuidores
        df_delta = montar_delta_por_mes(
            df_fatur=df_fatur,
            df_meta_mensal=df_meta_mensal,
            selected_dist=selected_dist,
            selected_produtos=selected_produtos,
            selected_ano=selected_ano
        )

        if not df_delta.empty:
            st.markdown("---")
            st.markdown("### 📈 Δ Kg por SKU (1–12)")

            skus_para_plotar = (
                selected_produtos 
                if selected_produtos 
                else sorted(df_delta['codigo_produto'].unique())
            )

            hoje = datetime.now()
            ano_atual = hoje.year
            mes_atual = hoje.month

            color_sequence = [
                "#FF5733","#33FF57","#3357FF","#FF33A1","#A133FF",
                "#33FFF5","#FF8C33","#8CFF33","#338CFF","#FF338C",
                "#33A1FF","#A1FF33","#FF3333","#33FF33","#3333FF",
                "#FF33FF","#33FFFF","#FFFF33","#D35400","#27AE60",
                "#2980B9","#8E44AD","#16A085","#F39C12","#C0392B"
            ]
            dist_list = sorted(selected_dist)
            dist_colors = { d: color_sequence[i % len(color_sequence)] for i, d in enumerate(dist_list) }

            cols = st.columns(4)
            col_index = 0

            for i, sku in enumerate(skus_para_plotar):
                df_sku = df_delta[df_delta['codigo_produto'] == sku].copy()
                if df_sku.empty:
                    continue

                fig_sku = go.Figure()
                global_y = []

                # === 1) loop que adiciona as traces “reais” de cada distribuidor ===
                for distrib in dist_list:
                    df_dist = df_sku[df_sku['nome_distribuidor'] == distrib].copy()
                    meses_completos = pd.DataFrame({'mes': list(range(1,13))})
                    df_dist = pd.merge(
                        meses_completos,
                        df_dist[['mes', 'Delta_Kg']],
                        on='mes', how='left'
                    ).fillna({'Delta_Kg': 0})

                    x_todos = df_dist['mes'].tolist()
                    y_todos = df_dist['Delta_Kg'].tolist()
                    global_y.extend(y_todos)

                    # montando apenas a trace sólida (Δ Kg “real”)
                    x_passados = [
                        m for m in x_todos
                        if not (selected_ano == ano_atual and m > mes_atual)
                    ]
                    y_passados = [
                        y for (m, y) in zip(x_todos, y_todos)
                        if not (selected_ano == ano_atual and m > mes_atual)
                    ]
                    if x_passados:
                        fig_sku.add_trace(
                            go.Scatter(
                                x=x_passados,
                                y=y_passados,
                                mode='lines+markers',
                                name=f"{distrib} (Real)",
                                line=dict(color=dist_colors[distrib], dash='solid'),
                                showlegend=False
                            )
                        )

                    # (projeção pontilhada já removida/comentada conforme pedido)

                # === 2) aqui que você insere o trecho de “shapes” para colorir o fundo ===

                # calcula limites para saber até onde pinta azul ou vermelho
                y_max = max(max(global_y), 0)
                y_min = min(min(global_y), 0)

                shapes = [
                    # área acima de y=0: azul suave
                    dict(
                        type="rect",
                        xref="paper", x0=0, x1=1,
                        yref="y",   y0=0, y1=y_max,
                        fillcolor="rgba(0, 0, 255, 0.02)",  # lightblue transparente
                        line=dict(width=0),
                        layer="below"
                    ),
                    # área abaixo de y=0: vermelho/rosa suave
                    dict(
                        type="rect",
                        xref="paper", x0=0, x1=1,
                        yref="y",   y0=y_min, y1=0,
                        fillcolor="rgba(255, 0, 0, 0.08)",  # lightpink transparente
                        line=dict(width=0),
                        layer="below"
                    )
                ]

                # === 3) atualização do layout, incluindo os shapes que acabamos de definir ===
                fig_sku.update_layout(
                    shapes=shapes,
                    title_text=f"SKU {sku} – Δ Kg por Distribuidor",
                    xaxis=dict(title="Mês"),
                    yaxis=dict(title="Δ Kg", tickformat=",.0f"),
                    width=300,
                    height=250,
                    margin=dict(l=30, r=10, t=30, b=30)
                )

                # === 4) exibição no Streamlit ===
                with cols[col_index]:
                    st.plotly_chart(
                        fig_sku,
                        use_container_width=True,
                        key=f"delta_{sku}_{i}"
                    )
                col_index = (col_index + 1) % 4


        else:
            st.markdown("---")
            st.markdown("### 📈 Δ Kg por SKU (sem dados disponíveis)")    
        # -------------------------------------------------------
        # Fim da seção “Δ Kg por SKU”
        # -------------------------------------------------------


    else:
        st.sidebar.info("1) Carregue a base de faturamento. 2) Carregue o arquivo de metas (CSV ou Excel). 3) Configure filtros e clique em ‘Calcular’.")

if __name__ == "__main__":
    main()
