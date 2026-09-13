# Fase 4 — Modelagem

```
python -m src.modeling.comparar_modelos    # baselines + 3 modelos, CV agrupada
python -m src.modeling.otimizar            # Optuna, 40 trials
python -m src.modeling.treinar_final       # treino final + abertura do teste
python -m src.evaluation.comparacao_justa  # modelo x baseline a sinalizacao igual
```

**Treino:** 1.559.528 alunos / 4.413 municípios · **Teste:** 292.324 alunos / 1.104 municípios
Zero municípios em comum. Taxa de alfabetização no teste: 60,0%.

> **Nota de versão.** Estes números substituem integralmente os de uma primeira rodada.
> Durante a Fase 5, a análise SHAP revelou um bug de ingestão que tratava 21,2% da base
> como cold start indevidamente (ver `reports/03_features.md`). Todo o pipeline foi
> reexecutado. O conjunto de teste foi, portanto, **aberto duas vezes** — a segunda por
> correção de dado auditada na fonte, não por insatisfação com a métrica. O registro fica
> aqui em vez de ser apagado.

---

## Por que o split é por município

Todas as features são municipais. Dois alunos do mesmo município são quase idênticos para o
modelo — diferem apenas por `rede`. Um split aleatório por linha colocaria o mesmo município
dos dois lados da divisão: o modelo memorizaria o resultado de 2024 daquele município e a
métrica ficaria otimista **sem que nada quebrasse no código**.

Pelo mesmo motivo a validação cruzada é `StratifiedGroupKFold`: `Group` para não repartir um
município entre treino e validação, `Stratified` para manter a proporção do alvo nos folds.

### O estrato tem três dimensões, e cada uma custou caro

`região × situação do histórico × porte`

- **situação do histórico** (completo / parcial / ausente) porque os três subgrupos têm
  informação disponível muito diferente e desempenho esperado muito diferente. Sem isso, a
  avaliação do subgrupo mais frágil vira sorteio.
- **porte** porque a estratificação conta **municípios** enquanto as métricas são calculadas
  sobre **alunos**. Na primeira rodada, sem essa dimensão, o teste ficou com 37,9% de cold
  start contra 23,1% da população — sortear alguns municípios grandes a mais desloca a
  composição em dezenas de pontos percentuais.

Sete estratos raros colapsaram (ex.: `Sudeste_ausente_GG`), preservando a dimensão que
importa para a avaliação por subgrupo.

Composição resultante:

| | completo | parcial | cold start |
|---|---|---|---|
| população | 76,9% | 21,2% | 1,9% |
| treino | 76,2% | 21,9% | 1,9% |
| **teste** | **80,7%** | **17,4%** | **1,9%** |

O cold start bate exato. Sobra desvio de 3,8 p.p. no histórico parcial: dentro da faixa de
maior porte os municípios ainda variam muito, e estratificação por município não controla
composição por aluno com precisão. É variação amostral registrada, não corrigida — ajustar a
partição depois de ver o resultado seria contaminar o teste.

---

## Baselines antes dos modelos

Sem régua, "AUC 0,66" não significa nada. O concorrente real não é o `DummyClassifier`, é a
regra que o gestor já usa hoje: **a taxa de alfabetização do município em 2023**.

## Comparação de modelos (CV agrupada, amostra de 300k)

| Modelo | AUC-ROC | AUC-PR risco | F1 risco | Tempo |
|---|---|---|---|---|
| RandomForest | 0,6556 | 0,5395 | 0,4146 | 164 s |
| **XGBoost** | 0,6543 | **0,5403** | **0,4214** | **83 s** |
| LogisticRegression | 0,6442 | 0,5256 | 0,4315 | 65 s |
| Baseline taxa municipal 2023 | 0,6278 | 0,5086 | 0,4822 | — |
| Baseline classe majoritária | 0,5000 | 0,4012 | 0,0000 | — |

**XGBoost escolhido**, apesar de o RandomForest ter AUC 0,0013 maior — diferença dentro do
ruído. O desempate foi AUC-PR de risco, F1 de risco e **o dobro da velocidade**, que importa
porque cada trial da busca roda 5 folds.

**A regressão logística chegar a 0,644 é em si um achado.** Se houvesse estrutura de
interação relevante entre as features, o boosting abriria vantagem clara. Abriu 0,010 — o
sinal disponível é essencialmente aditivo e está concentrado no nível educacional prévio do
município.

> Note que o baseline tem **F1 de risco maior que todos os modelos** (0,4822). Isso não é
> superioridade: no corte fixo de 0,5 ele sinaliza muito mais gente. A comparação honesta
> está mais adiante, a taxa de sinalização igual.

---

## Otimização de hiperparâmetros

Optuna (TPE) em vez de Grid: o espaço tem 8 dimensões e busca exaustiva é inviável.
`log=True` nos parâmetros multiplicativos — a diferença entre 0,01 e 0,02 importa tanto
quanto entre 0,2 e 0,4. `MedianPruner` com `trial.report` **fold a fold**, abandonando
combinações ruins antes de gastar os 5 folds.

**40 trials · 27,9 min · 18 trials podados (45%) · melhor AUC em CV = 0,6591**

| | AUC em CV |
|---|---|
| XGBoost padrão | 0,6543 |
| XGBoost otimizado | 0,6591 |
| **Ganho da busca** | **+0,0048** |

```
learning_rate=0,0221   max_depth=5   min_child_weight=11   n_estimators=391
subsample=0,693   colsample_bytree=0,557   reg_alpha=6,5e-06   reg_lambda=2,49
```

Árvores rasas, passo pequeno, muitas árvores, `colsample` agressivo: a busca convergiu para
**regularização e diversidade**, não para capacidade. Um espaço que permitia `max_depth` até
10 escolheu 5.

Comparação de magnitude que vale para a apresentação: a busca inteira comprou +0,0048,
enquanto a distância do baseline até o modelo no teste é +0,0262 — cinco vezes maior. E o
salto de qualidade real do projeto veio de auditar o dado, não de ajustar o modelo.

---

## Resultado no teste

| Métrica | Modelo final | Baseline municipal |
|---|---|---|
| **AUC-ROC** | **0,6633** | 0,6371 |
| AUC-PR (risco) | **0,5488** | 0,5234 |
| Acurácia | **0,6402** | 0,6226 |
| Brier | **0,2207** | 0,2280 |
| Precisão (risco) @0,5 | **0,5871** | 0,5395 |
| Recall (risco) @0,5 | 0,3362 | **0,3806** |
| F1 (risco) @0,5 | 0,4276 | **0,4463** |

**CV 0,6591 → teste 0,6633.** A estimativa fora da amostra ficou acima da validação cruzada,
depois de 40 trials de busca. Não há overfitting de seleção.

### Matriz de confusão (threshold 0,5)

| | previsto: não alfab. | previsto: alfabetizado |
|---|---|---|
| **real: não alfabetizado** | 39.285 | 77.559 |
| **real: alfabetizado** | 27.630 | 147.850 |

---

## Desempenho por qualidade do histórico

| | histórico completo | histórico parcial (Gold) | cold start |
|---|---|---|---|
| AUC-ROC | **0,6800** | 0,5735 | 0,5428 |
| AUC-PR risco | 0,5621 | 0,4538 | 0,5004 |
| Precisão risco | 0,5873 | 0,5579 | 0,5863 |
| **Recall risco @0,5** | 0,4113 | **0,0104** | 0,1753 |
| Brier | 0,2168 | 0,2359 | 0,2462 |
| n | 235.768 | 50.994 | 5.562 |

Três leituras, em ordem de importância:

**1. O modelo só é bom onde o histórico é completo.** AUC 0,68 contra 0,57 e 0,54. O
diferencial não é o município ter histórico, é ter `hist_proficiencia_media` — a feature
que a Fase 5 mostra ser a mais importante do modelo, e que só existe no microdado.

**2. No histórico parcial, o threshold de 0,5 torna o modelo inútil para detectar risco.**
Recall de **1%**. A explicação não é que o modelo não discrimina — a AUC de 0,5735 mostra
que discrimina um pouco. É que as probabilidades desse grupo ficam comprimidas **acima** de
0,5: sem proficiência, participação e dispersão, a imputação pela mediana empurra todos para
perto da média nacional, e quase ninguém cruza o corte.

> **Consequência operacional:** o corte de decisão precisa ser **por subgrupo**. Como o
> threshold é parâmetro de gestão e não exige retreinar, isso é ajuste de configuração, não
> de modelagem. Aplicar 0,5 uniformemente deixaria 51 mil crianças fora da triagem por um
> detalhe de calibração.

**3. O cold start real é pequeno e pouco informativo.** 5.562 alunos (1,9%), AUC 0,5428 —
pouco acima do acaso, como esperado quando restam apenas UF, rede e contexto socioeconômico.

---

## Threshold é decisão de negócio

| Threshold | Sinalizados | Precisão risco | Recall risco | F1 risco |
|---|---|---|---|---|
| 0,40 | 7,6% | 0,662 | 0,125 | 0,210 |
| 0,45 | 15,2% | 0,616 | 0,234 | 0,339 |
| **0,50** | **22,9%** | **0,587** | **0,336** | **0,428** |
| 0,55 | 38,9% | 0,533 | 0,519 | 0,525 |
| 0,60 | 54,8% | 0,500 | 0,685 | 0,578 |
| 0,65 | 64,8% | 0,480 | 0,778 | 0,594 |
| 0,70 | 72,3% | 0,464 | 0,839 | 0,598 |

Ajustável **sem retreinar**. Um programa com capacidade para 23% da rede opera em 0,50; um
que alcance metade opera em 0,60 e captura 69% das crianças em risco.

---

## O que a correção do bug de ingestão realmente rendeu

Vale registrar com honestidade, porque o resultado é contraintuitivo:

| | antes da correção | depois |
|---|---|---|
| Melhor AUC em CV (Optuna) | 0,6590 | 0,6591 |
| AUC no teste | 0,6570 | 0,6633 |

**Em validação cruzada, o ganho foi nulo.** Recuperar o histórico de 21,2% da base não moveu
a métrica de CV. A comparação no teste é favorável, mas as duas rodadas usam partições
diferentes — então ela indica, não prova.

A explicação mais provável: `uf` já carregava, em nível estadual, boa parte do que a taxa
municipal de São Paulo acrescentaria; e o que faltava de verdade (`hist_proficiencia_media`)
continua faltando, porque a Gold não tem essa coluna para SP.

**A correção continua certa**, por duas razões que não dependem da métrica: estávamos
descartando dado disponível, e a caracterização do cold start estava errada por um fator de
12 (23,1% contra 1,9% reais). Um relatório que atribuísse a um limite estrutural do dado
aquilo que era bug próprio seria pior que um relatório com AUC menor.

---

## Artefatos

| Arquivo | Conteúdo |
|---|---|
| `models/modelo_final.joblib` | Pipeline completo (imputação + encoding + XGBoost) |
| `reports/comparacao_modelos.{csv,json}` | Baselines e 3 modelos |
| `reports/melhores_hiperparametros.json` | Resultado da busca |
| `reports/optuna_historico.csv` | Trajetória dos 40 trials |
| `reports/analise_threshold.csv` | Trade-off completo |
| `reports/comparacao_justa.{csv,json}` | Modelo x baseline a sinalização igual |
| `reports/resultado_final.json` | Métricas do teste, geral e por subgrupo |
