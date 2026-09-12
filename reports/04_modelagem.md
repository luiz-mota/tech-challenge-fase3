# Fase 4 — Modelagem

```
python -m src.modeling.comparar_modelos   # baselines + 3 modelos, CV agrupada
python -m src.modeling.otimizar           # Optuna, 40 trials
python -m src.modeling.treinar_final      # treino final + abertura do teste
```

**Treino:** 1.408.725 alunos / 4.413 municípios · **Teste:** 443.127 alunos / 1.104 municípios
Zero municípios em comum.

---

## Por que o split é por município

Todas as features são municipais. Dois alunos do mesmo município são praticamente idênticos
para o modelo — diferem só por `rede`. Um split aleatório por linha colocaria o mesmo
município dos dois lados da divisão: o modelo memorizaria o resultado de 2024 daquele
município e a métrica ficaria otimista sem que nada de errado aparecesse no código.

Mesma lógica vale dentro do treino, por isso a validação cruzada é `StratifiedGroupKFold`:
`Group` para não repartir município entre treino e validação, `Stratified` para manter a
proporção do alvo em cada fold.

A estratificação do split é por **região × cold start**, colapsando estratos com menos de 10
municípios (`Sul_1` e `Centro-Oeste_1` caíram nessa regra — havia 1 município do
Centro-Oeste em cold start, impossível de repartir).

---

## Baselines primeiro

Sem régua, "AUC 0,65" não significa nada. Dois baselines:

| Baseline | O que é |
|---|---|
| Classe majoritária (`DummyClassifier`) | Piso absoluto |
| **Taxa de alfabetização do município em 2023** | O que um gestor faz hoje, sem modelo nenhum |

O segundo é o que importa: é o concorrente real do trabalho.

## Comparação de modelos (CV agrupada, amostra de 300k)

| Modelo | AUC-ROC | AUC-PR risco | F1 risco | Tempo |
|---|---|---|---|---|
| RandomForest | 0,6573 | 0,5405 | 0,3889 | 172 s |
| **XGBoost** | **0,6573** | **0,5417** | **0,4271** | **59 s** |
| LogisticRegression | 0,6566 | 0,5324 | 0,4182 | 65 s |
| Baseline taxa municipal 2023 | 0,6293 | 0,5159 | 0,4082 | — |
| Baseline classe majoritária | 0,5000 | 0,3990 | 0,0000 | — |

XGBoost escolhido: mesmo AUC do RandomForest, melhor F1 de risco, 3× mais rápido.

**A regressão logística empatar com os dois é em si um achado.** Se houvesse estrutura de
interação relevante entre as features, o boosting abriria vantagem. Não abriu — o sinal
disponível é essencialmente aditivo e está quase todo concentrado no histórico municipal.

---

## Otimização de hiperparâmetros

Optuna (TPE) em vez de Grid: o espaço tem 8 dimensões e busca exaustiva é inviável.
`log=True` nos parâmetros multiplicativos — a diferença entre 0,01 e 0,02 importa tanto
quanto entre 0,2 e 0,4. `MedianPruner` com `trial.report` **fold a fold**, para abandonar
combinações ruins antes de gastar os 5 folds.

**40 trials · 22,5 min · 29 trials podados (72%) · melhor AUC em CV = 0,6590**

| | AUC em CV |
|---|---|
| XGBoost padrão | 0,6573 |
| XGBoost otimizado | 0,6590 |
| **Ganho da busca** | **+0,0017** |

Uma busca inteira comprou 0,0017 de AUC. **Isso não é falha da busca — é o diagnóstico.** O
teto aqui é imposto pela informação disponível, não pelo ajuste do modelo. Compare com o
salto do baseline para o modelo (+0,039 no teste): a feature engineering valeu ~20× mais que
a otimização.

Os parâmetros escolhidos contam a mesma história:

```
learning_rate=0,0125   min_child_weight=163   max_depth=10
subsample=0,92   colsample_bytree=0,65   n_estimators=320
reg_alpha=7,6e-08   reg_lambda=0,0144
```

Passos pequenos e folhas grandes: a busca puxou para **regularização**, não para capacidade.
`max_depth=10` parece agressivo, mas com `min_child_weight=163` as árvores nunca chegam a
usar essa profundidade — a restrição que vale é o tamanho mínimo da folha.

---

## Resultado no teste

O conjunto de teste foi aberto aqui, **pela primeira e única vez**. Ele não participou de
nenhuma decisão: features, escolha de modelo e hiperparâmetros saíram exclusivamente de
validação cruzada sobre o treino.

| Métrica | Modelo final | Baseline municipal |
|---|---|---|
| AUC-ROC | **0,6570** | 0,6176 |
| AUC-PR (risco) | **0,5491** | 0,5134 |
| F1 (risco) | **0,4862** | 0,3824 |
| Precisão (risco) | 0,5659 | 0,5654 |
| **Recall (risco)** | **0,4262** | 0,2889 |
| Acurácia | 0,6300 | 0,6167 |
| Brier | **0,2241** | 0,2308 |

### Generalização

**CV 0,6590 → teste 0,6570.** Diferença de 0,002, depois de 40 trials de busca — que é
justamente onde o overfitting de seleção costuma aparecer. A estimativa é confiável.

### O ganho real não está no AUC

Com **precisão praticamente idêntica** (0,5659 vs 0,5654), o recall vai de 0,2889 para
0,4262. Em números absolutos, sobre as 182.016 crianças não alfabetizadas do teste:

| | crianças em risco identificadas |
|---|---|
| Regra municipal de hoje | 52.584 |
| Modelo | **77.568** |
| **Diferença** | **+24.984** |

**~25 mil crianças a mais encontradas, sem custo em precisão.** É esta a métrica de impacto,
não o AUC.

### Matriz de confusão (threshold 0,5)

| | previsto: não alfab. | previsto: alfabetizado |
|---|---|---|
| **real: não alfabetizado** | 77.568 | 104.448 |
| **real: alfabetizado** | 59.494 | 201.617 |

---

## Cold start: a limitação honesta

| | com histórico | cold start |
|---|---|---|
| AUC | 0,6892 | **0,5852** |
| F1 risco | 0,5026 | 0,4622 |
| Brier | 0,2144 | 0,2399 |
| n | 274.978 | 168.149 |

0,5852 é pouco acima do acaso. Para município que entra na avaliação sem histórico, sobram
UF, rede e contexto socioeconômico — e isso quase não discrimina.

Esta é a consequência direta do achado sobre `id_escola` (ver `reports/02_eda.md`): o ID da
escola é re-sorteado a cada ano, então **não existe histórico em nível de escola** para
suprir a falta do histórico municipal. Não é limitação do modelo; é informação que o dado
público não permite construir. Liga direto na recomendação de política pública.

### Viés de composição do teste

| | alunos em cold start | municípios em cold start |
|---|---|---|
| geral | 23,1% | 12,3% |
| treino | 18,5% | 12,3% |
| **teste** | **37,9%** | 12,1% |

A estratificação preservou a proporção de **municípios** sem histórico (12,3% → 12,1%), mas
não a de **alunos**: o sorteio mandou municípios grandes em cold start para o teste.

**O teste é mais difícil que a população real.** Como cold start tem AUC 0,5852 e ele pesa
37,9% no teste contra 23,1% na população, o 0,6570 é uma estimativa **conservadora**.

**Não refizemos o split.** Ajustar a partição depois de ver o resultado do teste é
exatamente como se contamina um conjunto de teste — o número deixaria de ser uma estimativa
honesta de generalização. A limitação fica registrada, e as métricas por subgrupo (acima)
são a forma correta de ler o resultado.

---

## Threshold é decisão de negócio

| Threshold | Sinalizados | Precisão risco | Recall risco | F1 risco |
|---|---|---|---|---|
| 0,40 | 7,2% | 0,663 | 0,116 | 0,197 |
| 0,45 | 12,0% | 0,634 | 0,186 | 0,287 |
| **0,50** | **30,9%** | **0,566** | **0,426** | **0,486** |
| 0,55 | 40,4% | 0,549 | 0,539 | 0,544 |
| 0,60 | 48,7% | 0,527 | 0,624 | 0,571 |
| 0,65 | 56,0% | 0,506 | 0,690 | 0,584 |
| 0,70 | 77,7% | 0,464 | 0,877 | 0,607 |

O corte é ajustável **sem retreinar**. Um programa com capacidade para atender 30% da rede
opera em 0,50; um que consiga atender metade opera em 0,60 e captura 62% das crianças em
risco. Para busca ativa ampla, 0,70 alcança 88%.

Nota: o salto entre 0,45 (12% sinalizados) e 0,50 (31%) reflete concentração da massa de
probabilidade — só existem ~6,5 mil perfis distintos de feature para 1,85M alunos, então
muitos alunos recebem exatamente a mesma previsão.

---

## Artefatos

| Arquivo | Conteúdo |
|---|---|
| `models/modelo_final.joblib` | Pipeline completo (imputação + encoding + XGBoost) |
| `reports/comparacao_modelos.{csv,json}` | Baselines e 3 modelos |
| `reports/melhores_hiperparametros.json` | Resultado da busca |
| `reports/optuna_historico.csv` | Trajetória dos 40 trials |
| `reports/analise_threshold.csv` | Trade-off completo |
| `reports/resultado_final.json` | Métricas do teste, geral e por subgrupo |
