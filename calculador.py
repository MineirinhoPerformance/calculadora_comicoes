import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
def load_metas_excel(metas_file):
    """
    Lê um arquivo Excel que pode ter várias abas, cada aba contendo:
      ETAPA | COD CLIENTE MINEIRINHO | COD DISTRIBUIDOR | NOME DISTRIBUIDOR | COD PROD | 01/01/2025 | 02/01/2025 | ...
    Retorna um DataFrame com colunas:
      ['nome_distribuidor', 'codigo_produto', 'data_dia', 'meta_kg_dia']
      Onde meta_kg_dia é float arredondado a duas casas decimais.
    """
    dfs_sheets = pd.read_excel(metas_file, sheet_name=None, dtype=str)
    lista_metas = []
    
    for sheet_name, df_raw in dfs_sheets.items():
        df = df_raw.copy()
        # Normaliza colunas
        df.columns = df.columns.str.strip()
        
        # Verifica colunas mínimas
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
        # Converte data_dia para datetime
        melt['data_dia'] = pd.to_datetime(melt['data_dia'], dayfirst=True, errors='coerce')
        # Limpa e converte meta_kg_dia para float
        melt['meta_kg_dia'] = (
            melt['meta_kg_dia']
            .astype(str)
            .str.replace('.', '', regex=False)   # remove ponto de milhar
            .str.replace(',', '.', regex=False)  # vírgula → ponto
        )
        melt['meta_kg_dia'] = pd.to_numeric(melt['meta_kg_dia'], errors='coerce').fillna(0.0)
        # ** Ajuste: arredonda a duas casas decimais **
        melt['meta_kg_dia'] = melt['meta_kg_dia'].round(2)
        # Padroniza nomes
        melt.rename(columns={'NOME DISTRIBUIDOR': 'nome_distribuidor', 
                             'COD PROD': 'codigo_produto'}, inplace=True)
        
        lista_metas.append(melt[['nome_distribuidor', 'codigo_produto', 'data_dia', 'meta_kg_dia']])
    
    if not lista_metas:
        return pd.DataFrame(columns=['nome_distribuidor','codigo_produto','data_dia','meta_kg_dia'])
    
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
      Onde 'meta_kg_mes' é soma dos valores já arredondados a duas casas.
    """
    df = df_metas.copy()
    df['ano'] = df['data_dia'].dt.year
    df['mes'] = df['data_dia'].dt.month
    df_mes = (
        df
        .groupby(['nome_distribuidor','codigo_produto','ano','mes'], as_index=False)
        .agg(meta_kg_mes=('meta_kg_dia','sum'))
    )
    # Garante strings
    df_mes['nome_distribuidor'] = df_mes['nome_distribuidor'].astype(str)
    df_mes['codigo_produto']    = df_mes['codigo_produto'].astype(str)
    # Arredonda de novo (caso haja soma que precise de ajuste)
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
                Total_Kg_Ant=('total kg','sum'),
                Faturamento_Ant=('total df','sum')
            )
        )
        df_prev_group['Preco_Kg_Ant'] = df_prev_group.apply(
            lambda r: (r['Faturamento_Ant'] / r['Total_Kg_Ant']) if r['Total_Kg_Ant'] > 0 else 0,
            axis=1
        )
        
        # 3) Faz merge entre corrente e anterior
        df_merge = pd.merge(
            df_current, df_prev_group,
            on=['nome_distribuidor','codigo_produto'],
            how='left'
        ).fillna({'Total_Kg_Ant':0,'Faturamento_Ant':0,'Preco_Kg_Ant':0})
        
        # 4) Calcula Δ
        df_merge['Delta_Kg'] = df_merge['Total_Kg_Mes'] - df_merge['Total_Kg_Ant']
        df_merge['Delta_R']  = df_merge['Faturamento_Mes'] - df_merge['Faturamento_Ant']
        
        # 5) Agrega meta mensal (ano=selected_ano, mes=mes)
        df_meta_mes_corrente = df_meta_mensal[
            (df_meta_mensal['ano'] == selected_ano) &
            (df_meta_mensal['mes'] == mes)
        ][['nome_distribuidor','codigo_produto','meta_kg_mes']].copy()
        df_merge = pd.merge(
            df_merge,
            df_meta_mes_corrente,
            on=['nome_distribuidor','codigo_produto'],
            how='left'
        ).fillna({'meta_kg_mes':0.0})
        df_merge.rename(columns={'meta_kg_mes':'meta_kg'}, inplace=True)
        
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
            return pd.Series({'Kg_T1':kg_t1,'Kg_T2':kg_t2,'Kg_T3':kg_t3})
        
        if df_merge.empty:
            df_merge[['Kg_T1','Kg_T2','Kg_T3']] = 0, 0, 0
        else:
            faixas = df_merge.apply(calcular_faixas, axis=1)
            faixas.columns = [c.strip() for c in faixas.columns]
            df_merge[['Kg_T1','Kg_T2','Kg_T3']] = faixas[['Kg_T1','Kg_T2','Kg_T3']]
        
        # 7) Calcula valores e comissões
        df_merge['Val_T1'] = df_merge['Kg_T1'] * df_merge['Preco_Kg_Mes']
        df_merge['Val_T2'] = df_merge['Kg_T2'] * df_merge['Preco_Kg_Mes']
        df_merge['Val_T3'] = df_merge['Kg_T3'] * df_merge['Preco_Kg_Mes']
        
        df_merge['Com_T1'] = df_merge['Val_T1'] * (pct1/100)
        df_merge['Com_T2'] = df_merge['Val_T2'] * (pct2/100)
        df_merge['Com_T3'] = df_merge['Val_T3'] * (pct3/100)
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
        df_annual = pd.DataFrame(columns=['nome_distribuidor','Comissao_R$','mes'])
    return df_annual

# -------------------------------------------------------
# Fluxo principal
# -------------------------------------------------------
def main():
    st.title("📊 Calculadora de Comissões por KG")

    st.markdown("""
    Este aplicativo calcula comissões mensais e anual com base em:
    1. **Base de faturamento** (Excel carregado pelo usuário);
    2. **Metas diárias** (segundo Excel, que pode ter várias abas), onde cada célula representa a meta de KG daquele dia, por distribuidor e produto.
    """)
    
    st.sidebar.header("📁 Importar dados")
    uploaded_fat = st.sidebar.file_uploader(
        "1) Carregue aqui o Excel da base de faturamento",
        type=["xlsx", "xls"], key="fat"
    )
    uploaded_meta = st.sidebar.file_uploader(
        "2) Carregue aqui o Excel de metas diárias (pode ter várias abas)",
        type=["xlsx", "xls"], key="meta"
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
            df_metas_diarias = load_metas_excel(uploaded_meta)
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
        ano_selecionado  = st.selectbox("Ano de análise", anos, index=(len(anos)-1) if anos else 0) if anos else None
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
            df_display['Meta Kg (mês)']   = df_display['meta_kg'].apply(lambda x: f"{x:,.2f}")
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
            - **Meta Kg (mês)**: soma das metas diárias (do Excel de metas) para aquele distribuidor/sku no mês selecionado, já arredondadas a duas casas decimais.  
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
        totais_merge['Meta Kg (mês)']   = totais_merge['Sum_Meta_Kg'].apply(lambda x: f"{x:,.2f}")
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
        col3.metric("Meta Kg (Total)", f"{total_meta_kg_all:,.2f}")
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
        #  Gráfico Anual de Comissões por Mês e Distribuidor
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("**Gráfico Anual de Comissões por Mês e Distribuidor**")
        
        df_annual = calcular_comissoes_mensais(
            df_fatur, df_meta_mensal,
            selected_dist, selected_produtos,
            pct1, pct2, pct3,
            selected_ano
        )
        if not df_annual.empty:
            df_annual['mes_str'] = df_annual['mes'].apply(lambda x: f"{x:02d}")
            df_annual.rename(columns={'nome_distribuidor':'Distribuidor','Comissao_R$':'Comissao_Num'}, inplace=True)
            
            fig_annual = px.bar(
                df_annual,
                x='mes_str',
                y='Comissao_Num',
                color='Distribuidor',
                color_discrete_map=dist_colors,
                labels={'mes_str':'Mês','Comissao_Num':'Comissão (R$)'},
                title=None
            )
            df_total_mes = df_annual.groupby('mes_str', as_index=False).agg(Total_Mes=('Comissao_Num','sum'))
            for idx, row in df_total_mes.iterrows():
                fig_annual.add_annotation(
                    x=row['mes_str'],
                    y=row['Total_Mes'],
                    text=f"R$ {row['Total_Mes']:,.2f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=14,color="black"),
                    bgcolor="rgba(255,255,255,0.7)"
                )
            fig_annual.update_layout(
                barmode='stack',
                uniformtext_minsize=12, uniformtext_mode='hide',
                yaxis_tickformat=",.2f",
                margin=dict(t=20,b=20,l=40,r=20),
                xaxis_title="Mês", yaxis_title="Comissão (R$)"
            )
            st.plotly_chart(fig_annual, use_container_width=True)
        else:
            st.info("Não há dados de comissão anual para os filtros atuais.")
        
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
        st.sidebar.info("1) Carregue o Excel de faturamento. 2) Carregue o Excel de metas. 3) Configure filtros e clique em ‘Calcular’.")

if __name__ == "__main__":
    main()
