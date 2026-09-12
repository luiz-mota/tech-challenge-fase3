# Fase 1 — Ingestão

Extração reproduzível via `python -m src.ingestion.extract_bigquery`.
Custo medido: **249 MB processados (0,02% da cota gratuita mensal do BigQuery) — R$ 0,00.**

## Fontes extraídas

| Arquivo | Linhas | Origem | Papel no projeto |
|---|---:|---|---|
| `alunos` | 3.867.999 | `basedosdados.br_inep_avaliacao_alfabetizacao.alunos` | Grão do modelo (1 linha = 1 aluno avaliado) |
| `gold_indicador_municipio` | 10.896 | Camada Gold da Fase 2 | Desempenho municipal por ano |
| `gold_metas_vs_resultados` | 5.500 | Camada Gold da Fase 2 | Metas 2024-2030 e `gap_meta_2030` |
| `gold_evolucao_temporal` | 10.896 | Camada Gold da Fase 2 | Série por município/ano |
| `idhm_municipio` | 5.565 | `mundo_onu_adh.municipio` (censo 2010) | Contexto socioeconômico estrutural |
| `pib_populacao_municipio` | 5.570 | `br_ibge_pib` ⋈ `br_ibge_populacao` (2023) | Contexto econômico contemporâneo |
| `censo_escolar_municipio` | 5.570 | `br_inep_censo_escolar.escola` (2023), agregado | Infraestrutura escolar do município |

## Decisões de extração

**Extração sem filtro de presença.** A camada raw é fiel à origem; os filtros ficam no
pré-processamento. Isso preserva as linhas de alunos ausentes, necessárias para documentar
o segundo vazamento (abaixo) com evidência.

**`proficiencia` extraída, mas não é feature.** Ela entra na raw apenas para a EDA
comprovar a relação com o alvo. É descartada antes da modelagem.

**Colunas descartadas na origem:** `id_aluno` (identificador puro, sem sinal) e `serie`
(constante — 1 único valor em toda a base).

**Censo Escolar agregado por município, não por escola.** O `id_escola` dos microdados de
alfabetização é mascarado (códigos fictícios, ex: `60000113`), incompatível com o código
INEP do Censo Escolar. O join por escola é impossível; por município é exato. Filtramos a
escolas em funcionamento que ofertam anos iniciais do fundamental — o universo que
efetivamente atende a criança avaliada.

**Recorte de 22 das 230 colunas do IDHM.** Índices sintéticos, renda, pobreza, escolaridade
adulta (proxy de capital cultural domiciliar) e saneamento — dimensões com literatura
consolidada ligando-as a desempenho escolar.

## Dois vazamentos confirmados empiricamente

Ambos estão travados como testes em `tests/test_raw_data.py`.

**1. `proficiencia` é o alvo antes do corte.** O Indicador Criança Alfabetizada é definido
como proficiência ≥ 743 na escala Saeb. Verificado: `alfabetizado == (proficiencia >= 743)`
em **100,0000%** das 3.354.661 linhas válidas, nas duas direções. `proficiencia` nunca pode
ser feature — um modelo que a receba apenas reaprende o `if`.

**2. Aluno ausente é sempre "não alfabetizado".** Todos os 512.153 alunos com
`presenca = '0'` têm `alfabetizado = '0'` e proficiência nula — codificação administrativa,
não medição. `presenca` prevê o alvo perfeitamente em 13% das linhas, então vira **filtro**,
nunca feature.

## Universo de modelagem resultante

Após `presenca = '1' AND preenchimento_caderno = '1'`:

- **3.354.661 alunos** (de 3.867.999)
- **59,2% alfabetizados** — alvo balanceado, sem necessidade de tratamento de
  desbalanceamento e sem o risco clássico de acurácia enganosa

## Limitações já identificadas

- **IDHM é de 2010.** O Atlas do Desenvolvimento Humano só publica em anos censitários; é
  o dado mais recente disponível. Serve como proxy estrutural (desenvolvimento municipal é
  persistente), mas tem 14 anos de defasagem em relação ao ano previsto. PIB e população de
  2023 compensam parcialmente.
- **Sem atributos individuais do aluno.** Os microdados são anonimizados sob LGPD — não há
  sexo, raça/cor, idade ou condição socioeconômica. Todo o sinal disponível é contextual
  (escola e município).
