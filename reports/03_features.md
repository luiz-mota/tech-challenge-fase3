# Fase 3 — Feature Engineering

`python -m src.preprocessing.build_features` → `data/processed/dataset_modelagem.parquet`
**1.851.852 linhas × 61 colunas** (alunos avaliados em 2024), 5.517 municípios.

Desenho: **features de 2023 → alvo de 2024**. Nenhuma feature usa informação do ano que
está sendo previsto. Isso não é só higiene metodológica — o caso de uso é triagem *antes*
da avaliação, quando o resultado de 2024 ainda não existe.

---

## ⚠️ Vazamento encontrado na camada Gold

`gold_metas_vs_resultados` guarda **o ano mais recente por município** — que para 5.448 dos
5.500 registros é **2024**. Isso contamina duas colunas:

| Coluna | Situação |
|---|---|
| `taxa_alfabetizacao` | É o resultado de 2024 — exatamente o alvo |
| `gap_meta_2030` | Derivado dele (`meta_2030 − taxa_alfabetizacao`) |
| `meta_alfabetizacao_*` | ✅ **Seguras** — pactuadas por política, fixas por município (1 valor único por município, verificado) |

**Correção:** as duas primeiras foram descartadas. O gap foi **recalculado de forma limpa**:

```
gap_meta_2024 = meta_alfabetizacao_2024 − taxa_alfabetização do município em 2023
```

Ou seja, "quão distante o município estava da sua meta de 2024, medido só com o que se sabia
ao entrar em 2024". Mesma intenção analítica, sem contaminação.

Esse é o terceiro vazamento distinto encontrado no projeto — os dois primeiros estão em
`reports/01_ingestao.md`. Todos estão travados como teste em `tests/`.

---

## Inventário de features

### Histórico municipal de 2023 (o preditor mais forte)
| Feature | Racional |
|---|---|
| `hist_taxa_alfabetizacao` | Correlação 0,70 entre anos — o melhor preditor isolado |
| `hist_proficiencia_media` | Captura *quão longe* do corte, não só o passa/não-passa |
| `hist_n_avaliados`, `hist_n_escolas` | Porte da rede avaliada |
| `hist_desvio_entre_escolas` | **Desigualdade interna**: dois municípios com a mesma média, um homogêneo e outro heterogêneo, pedem intervenções diferentes |
| `hist_taxa_participacao` | Município que não leva a criança à prova sinaliza fragilidade de gestão — não capturado pela taxa de acerto |

### Metas pactuadas (conhecidas de antemão)
`meta_alfabetizacao_2024`, `meta_alfabetizacao_2026`, `gap_meta_2024` (recalculado).

`meta_alfabetizacao_2030` foi **descartada**: vale 80 para todo município — a meta de 2030 é
nacional e uniforme, então não diferencia ninguém.

### Contexto socioeconômico (IDHM, censo 2010)
IDHM e seus componentes, renda per capita, Gini, pobreza (geral e infantil), analfabetismo
adulto, expectativa de anos de estudo, frequência escolar, crianças fora da escola,
saneamento, proporção urbana.

### Economia municipal (2023)
`pib_per_capita`, `populacao`, e composição setorial (`part_va_agropecuaria`,
`part_va_industria`, `part_va_servicos`).

### Infraestrutura escolar (Censo Escolar 2023, agregado por município)
Proporção de escolas com biblioteca, laboratório, internet, banda larga, água potável,
esgoto, energia, alimentação, quadra, acessibilidade; proporção rural; média de salas; e as
razões `censo_alunos_por_turma` e `censo_alunos_por_docente`.

### Contexto direto e flags
`rede` (municipal/estadual/privada), `uf`, `regiao`, `sem_historico_municipal`.

---

## Decisões de encoding

| Tipo | Tratamento | Por quê |
|---|---|---|
| Numéricas | `SimpleImputer(median)` + `StandardScaler` opcional | Mediana porque as distribuições municipais são assimétricas (PIB per capita, população) e a média seria puxada pelos extremos. Scaling é desligável: árvores não precisam |
| Categóricas | `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')` | Cardinalidade baixa (3/5/27) torna One-Hot viável e preserva a ausência de ordem. `handle_unknown='ignore'` porque um fold pode conter UF ausente do treino |
| `sem_historico_municipal` | passthrough | Já é 0/1 e carrega significado próprio |
| `id_municipio` | **removida das features** | Serve só para agrupar no cross-validation |

**Não usamos Target Encoding.** Ele calcula a média do alvo por categoria e é a porta de
entrada clássica de vazamento; com 27 UFs, o ganho não compensa o risco.

---

## Por que tudo dentro de um `Pipeline`

Imputação e scaling são *aprendidos* dos dados (mediana, média, desvio). Ajustados na base
inteira, a estatística do conjunto de validação vaza para o treino e a métrica fica
otimista. Dentro do `Pipeline`, o `fit` acontece uma vez por fold, automaticamente — e o
mesmo objeto serializa para produção.

Isso é verificado pelo teste `test_pipeline_ajusta_imputacao_apenas_no_treino`, que confere
que a mediana aprendida é a do treino, não a da base completa.

---

## Guardas contra features mortas

O build descarta automaticamente colunas sem variação, logando o que caiu. Dois casos reais
apareceram:

- **100% nulas na origem:** valor adicionado setorial de 2023 (o IBGE só publicou o PIB
  total; puxamos a composição de 2021 como proxy estrutural) e `quantidade_computador_aluno`
  (nula em todos os anos, 2019-2024, apesar de existir no schema).
- **Constantes:** `meta_alfabetizacao_2030`.

Sem essa guarda, elas sobreviveriam à imputação disfarçadas de feature e poluiriam o SHAP.

---

## Nulos remanescentes (todos legítimos)

| Faixa | Colunas | Causa |
|---|---|---|
| ~23-26% | `hist_*`, `gap_meta_2024` | Cold start: municípios que entraram na avaliação em 2024 |
| ~3-5% | `meta_alfabetizacao_*` | Municípios sem metas pactuadas |
| ~0,1% | IDHM | 5 municípios criados após o censo de 2010 |

Todos são imputados **dentro do pipeline**, por fold. O cold start ainda ganha flag própria
e será avaliado separadamente na Fase 4.

---

## Verificação

`pytest tests/` → **20/20**. Os testes de pré-processamento travam: nenhuma coluna proibida
no dataset, nenhuma feature com |r| > 0,9 com o alvo, imputação ajustada só no treino, e o
pipeline tolerando categoria inédita no fold de validação.
