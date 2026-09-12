# Predição e Inteligência Analítica para Alfabetização no Brasil

Tech Challenge — Fase 3 (Pós-Tech FIAP · AI Scientist). Modelo supervisionado de
classificação binária para prever se um aluno do 2º ano do ensino fundamental será
considerado **alfabetizado**, a partir dos microdados do Indicador Criança Alfabetizada
integrados na camada Gold construída na Fase 2.

> 🚧 **Em desenvolvimento.** Este README é preenchido incrementalmente ao longo das fases
> do projeto. Seções marcadas com _(pendente)_ ainda não foram escritas.

---

## 1. Contexto do problema

O **Compromisso Nacional Criança Alfabetizada** (Decreto nº 11.556/2023) estabelece metas
de alfabetização para cada município brasileiro até 2030, medidas por avaliações estaduais
aplicadas ao final do 2º ano do ensino fundamental. A partir da Pesquisa Alfabetiza Brasil
(INEP, 2023), definiu-se o **ponto de corte de 743 pontos na escala Saeb** como o patamar
a partir do qual uma criança é considerada alfabetizada.

Gestores públicos precisam antecipar risco educacional — identificar onde a alfabetização
tende a não avançar **antes** do resultado da próxima avaliação, para direcionar apoio
técnico e financeiro com antecedência.

## 2. Objetivo analítico

Prever, para cada aluno avaliado em 2024, a probabilidade de ser considerado alfabetizado,
usando **exclusivamente informação disponível antes da avaliação** (histórico de 2023 da
escola e do município + contexto socioeconômico municipal).

A saída alimenta três usos: ranking de risco por município, previsão de municípios que não
atingirão a meta, e agrupamento de territórios com padrões semelhantes.

## 3. Descrição da base utilizada

_(pendente — Fase 2 do projeto)_

**Origem:** camada Gold do Tech Challenge da Fase 2 (BigQuery), construída sobre o dataset
público `br_inep_avaliacao_alfabetizacao` da [Base dos Dados](https://basedosdados.org/),
enriquecida com fontes socioeconômicas municipais.

## 4. Etapas de modelagem

_(pendente — Fases 3 e 4)_

## 5. Escolha do algoritmo

_(pendente — Fase 4)_

## 6. Métricas de avaliação

_(pendente — Fase 4)_

## 7. Interpretação dos resultados

_(pendente — Fase 5)_

## 8. Insights encontrados

_(pendente — Fases 5 e 6)_

## 9. Limitações do projeto

_(pendente — consolidado ao final)_

Ponto já identificado: os microdados são anonimizados sob LGPD e **não contêm atributos
individuais do aluno** (sexo, raça/cor, idade, condição socioeconômica). O sinal disponível
é contextual — escola e município —, o que delimita o que a predição individual pode
alcançar. Isso é tratado como achado analítico, não como falha de modelagem.

## 10. Aplicação prática para políticas públicas

_(pendente — Fase 6)_

## 11. Possíveis evoluções futuras

_(pendente — consolidado ao final)_

---

## Como rodar

```bash
pip install -r requirements.txt
cp .env.example .env    # preencher com o caminho local da service account
python -m src.ingestion.extract_bigquery
```

## Estrutura do repositório

```
data/            # dados brutos e processados (não versionados — reproduzíveis via src/ingestion)
notebooks/       # análise exploratória e aplicação estratégica
src/
├── ingestion/      # extração do BigQuery (microdados + Gold + enriquecimento)
├── preprocessing/  # filtros, feature engineering, ColumnTransformer
├── modeling/       # baseline, treino, otimização de hiperparâmetros
├── evaluation/     # métricas, SHAP, análise de threshold
└── visualization/  # gráficos para relatórios e vídeo executivo
reports/         # achados escritos de cada fase
images/          # gráficos exportados
tests/           # testes das funções de pré-processamento
```
