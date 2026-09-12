"""Queries de extração — uma constante por fonte.

Separadas do executor para ficarem legíveis e revisáveis isoladamente (é aqui que
moram as decisões analíticas que precisam ser justificadas no README).
"""

BD = "basedosdados"

# Ano de referência das features. O desenho do projeto usa o histórico de 2023
# para prever os alunos avaliados em 2024 — assim nenhuma feature carrega
# informação do próprio ano que está sendo previsto.
ANO_FEATURES = 2023
ANO_ALVO = 2024

# Censo do IDHM. O Atlas do Desenvolvimento Humano só publica em anos
# censitários (1991/2000/2010); 2010 é o mais recente disponível.
ANO_IDHM = 2010


# ---------------------------------------------------------------------------
# 1. Microdados por aluno — grão do modelo
# ---------------------------------------------------------------------------
# Extraído SEM filtro de presença de propósito: a camada raw precisa ser fiel à
# origem, e a EDA usa as linhas de alunos ausentes para documentar o segundo
# vazamento (ausente => alfabetizado=0 por construção). O filtro é aplicado
# depois, em src/preprocessing.
#
# `proficiencia` também vem junto, mas NÃO é feature: alfabetizado é exatamente
# proficiencia >= 743. Ela existe aqui só para a EDA comprovar essa relação.
# Colunas descartadas: id_aluno (identificador, sem sinal) e serie (constante).
ALUNOS = f"""
SELECT
    ano,
    id_municipio,
    id_escola,
    rede,
    caderno,
    presenca,
    preenchimento_caderno,
    alfabetizado,
    proficiencia,
    peso_aluno
FROM `{BD}.br_inep_avaliacao_alfabetizacao.alunos`
"""


# ---------------------------------------------------------------------------
# 2. Camada Gold da Fase 2 — contexto municipal
# ---------------------------------------------------------------------------
GOLD_INDICADOR_MUNICIPIO = "SELECT * FROM `{project}.{dataset}.indicador_municipio`"
GOLD_METAS_VS_RESULTADOS = "SELECT * FROM `{project}.{dataset}.metas_vs_resultados`"
GOLD_EVOLUCAO_TEMPORAL = "SELECT * FROM `{project}.{dataset}.evolucao_temporal`"


# ---------------------------------------------------------------------------
# 3. IDHM / Atlas do Desenvolvimento Humano (município, censo 2010)
# ---------------------------------------------------------------------------
# Recorte das 230 colunas disponíveis: índices sintéticos, renda, pobreza,
# indicadores educacionais dos adultos (proxy de capital cultural domiciliar) e
# saneamento — dimensões com literatura consolidada ligando-as a desempenho
# escolar. Puxar as 230 colunas inteiras só aumentaria ruído e custo.
IDHM = f"""
SELECT
    id_municipio,
    idhm,
    idhm_e,
    idhm_l,
    idhm_r,
    renda_pc,
    indice_gini,
    prop_pobreza,
    prop_pobreza_criancas,
    taxa_analfabetismo_15_mais,
    taxa_analfabetismo_25_mais,
    expectativa_anos_estudo,
    taxa_freq_liquida_fundamental,
    taxa_criancas_fora_escola_6_14,
    taxa_criancas_dom_sem_fund,
    taxa_agua_encanada,
    taxa_coleta_lixo,
    taxa_energia_eletrica,
    taxa_agua_esgoto_inadequados,
    taxa_mulheres_com_filho_10_14,
    populacao_urbana,
    populacao_rural
FROM `{BD}.mundo_onu_adh.municipio`
WHERE ano = {ANO_IDHM}
"""


# ---------------------------------------------------------------------------
# 4. PIB e população municipais (contexto econômico contemporâneo)
# ---------------------------------------------------------------------------
# O IDHM é de 2010; PIB e população de {ANO_FEATURES} corrigem parte dessa
# defasagem. O PIB per capita é calculado no join, não aqui, para manter cada
# query com responsabilidade única.
#
# O desmembramento setorial (valor adicionado por setor) só existe até 2021 na
# fonte — em 2022/2023 o IBGE publicou apenas o PIB total. Como a composição
# setorial de um município muda devagar, puxamos a de ANO_VA como proxy
# estrutural, num LEFT JOIN separado para não perder os municípios.
ANO_VA = 2021

PIB_POPULACAO = f"""
SELECT
    pib.id_municipio,
    pop.sigla_uf,
    pib.pib,
    pop.populacao,
    va.va_agropecuaria,
    va.va_industria,
    va.va_servicos,
    va.va_adespss
FROM `{BD}.br_ibge_pib.municipio` AS pib
INNER JOIN `{BD}.br_ibge_populacao.municipio` AS pop
    ON pib.id_municipio = pop.id_municipio AND pib.ano = pop.ano
LEFT JOIN (
    SELECT id_municipio, va_agropecuaria, va_industria, va_servicos, va_adespss
    FROM `{BD}.br_ibge_pib.municipio`
    WHERE ano = {ANO_VA}
) AS va ON va.id_municipio = pib.id_municipio
WHERE pib.ano = {ANO_FEATURES}
"""


# ---------------------------------------------------------------------------
# 5. Censo Escolar agregado por município
# ---------------------------------------------------------------------------
# Agregamos por município (e não por escola) porque `id_escola` nos microdados
# de alfabetização é mascarado — códigos fictícios que não batem com o código
# INEP do Censo Escolar. O join por escola é impossível; por município é exato.
#
# Recorte: só escolas que ofertam anos iniciais do fundamental e estão em
# funcionamento — é o universo que efetivamente atende a criança avaliada.
# As flags binárias viram proporção de escolas do município que têm o recurso.
CENSO_ESCOLAR_MUNICIPIO = f"""
SELECT
    id_municipio,
    COUNT(*)                                          AS censo_n_escolas,
    AVG(biblioteca)                                   AS censo_prop_biblioteca,
    AVG(sala_leitura)                                 AS censo_prop_sala_leitura,
    AVG(laboratorio_informatica)                      AS censo_prop_lab_informatica,
    AVG(internet)                                     AS censo_prop_internet,
    AVG(banda_larga)                                  AS censo_prop_banda_larga,
    AVG(agua_potavel)                                 AS censo_prop_agua_potavel,
    AVG(esgoto_rede_publica)                          AS censo_prop_esgoto_publico,
    AVG(energia_rede_publica)                         AS censo_prop_energia_publica,
    AVG(alimentacao)                                  AS censo_prop_alimentacao,
    AVG(quadra_esportes)                              AS censo_prop_quadra,
    AVG(banheiro_pne)                                 AS censo_prop_banheiro_pne,
    AVG(acessibilidade_rampas)                        AS censo_prop_rampas,
    AVG(CASE WHEN tipo_localizacao = '2' THEN 1 ELSE 0 END) AS censo_prop_rural,
    AVG(quantidade_sala_utilizada)                    AS censo_media_salas,
    -- quantidade_computador_aluno foi descartada: está 100% nula em todos os anos
    -- da fonte (2019-2024), apesar de existir no schema.
    SUM(quantidade_matricula_fundamental_anos_iniciais) AS censo_matriculas_anos_iniciais,
    SUM(quantidade_docente_fundamental_anos_iniciais)   AS censo_docentes_anos_iniciais,
    SUM(quantidade_turma_fundamental_anos_iniciais)     AS censo_turmas_anos_iniciais
FROM `{BD}.br_inep_censo_escolar.escola`
WHERE ano = {ANO_FEATURES}
  -- flags do Censo Escolar são INTEGER (0/1), não BOOLEAN
  AND etapa_ensino_fundamental_anos_iniciais = 1
  AND tipo_situacao_funcionamento = '1'
GROUP BY id_municipio
"""
