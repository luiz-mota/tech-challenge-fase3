# Fase 2 — Análise Exploratória

Notebook: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)

A EDA foi conduzida para **decidir**, não para descrever. Cada seção do notebook termina
numa decisão de modelagem. Este documento consolida os achados e o que cada um implica.

---

## O achado central: `id_escola` não é rastreável entre anos

O identificador de escola nos microdados é mascarado ("códigos fictícios", conforme o
dicionário da fonte). Testamos se ele ainda assim identifica a mesma escola em 2023 e 2024 —
escolas não mudam de município, então o município deveria coincidir em ~100% dos casos.

| Nível | Observado | Esperado ao acaso | Se o ID fosse estável |
|---|---:|---:|---:|
| município | **2,40%** | 0,18% | ~100% |
| UF | **80,63%** | 6,63% | ~100% |

**97,6% dos "pares" entre anos são escolas diferentes.** A máscara é alocada em blocos por
estado (daí a UF coincidir em 80%) e embaralhada dentro do estado a cada ano.

Confirmação independente — correlação da taxa de alfabetização entre 2023 e 2024:

| Unidade | Correlação | n |
|---|---:|---:|
| Escola (ID mascarado) | **0,25** | 18.756 |
| Município (código IBGE real) | **0,70** | 4.277 |

E o efeito prático, medido como AUC ao prever os alunos de 2024:

| Feature | AUC |
|---|---:|
| Taxa do **município** em 2023 | **0,6280** |
| Taxa da **escola** em 2023 | 0,5521 ← pior, por ser ~98% ruído |

### Por que isso importa

A seção 4 do notebook mostra que a escola **é** a unidade relevante: 17,7% da variância do
resultado fica entre escolas, e escolas do mesmo município divergem em média 13,4 p.p. O
sinal existe — mas é inalcançável entre anos.

**Consequência para o projeto:** nenhuma feature histórica por escola é viável. A modelagem
fica restrita a features municipais.

**Consequência para política pública:** publicar um identificador de escola pseudonimizado
porém **estável entre anos** — sem expor a identidade da escola — habilitaria monitoramento
longitudinal e modelos de risco substancialmente mais precisos. É uma recomendação concreta
que sai desta análise.

---

## Demais achados

### Desigualdade territorial é enorme
Entre a UF de menor e a de maior taxa, **49,4 pontos percentuais** de diferença (36,0% a
85,3% em 2024). Rede estadual tem taxa consistentemente maior que a municipal
(62,5% vs 59,4% em 2024).

### Enriquecimento cobre tudo, mas nenhuma variável isolada resolve
As três fontes cobrem ~100% dos alunos de 2024. As correlações com a taxa municipal são
coerentes em sinal e teoricamente plausíveis, porém **moderadas** (|r| ≤ 0,27):

| Variável | r |
|---|---:|
| `expectativa_anos_estudo` | +0,264 |
| `censo_prop_quadra` | +0,257 |
| `prop_pobreza` | −0,250 |
| `taxa_agua_esgoto_inadequados` | −0,241 |
| `idhm_e` (IDHM educação) | +0,240 |

O modelo precisará combiná-las; não há atalho.

### Cold start é relevante

> ⚠️ **Este achado foi posteriormente revisado.** A conclusão abaixo estava correta em
> relação ao que havia sido extraído, mas **errada em relação ao que existia na fonte**. A
> análise SHAP da Fase 5 revelou que 92,4% desse "cold start" era São Paulo, cujo histórico
> de 2023 estava disponível na camada Gold e não havia sido usado — falha de ingestão nossa.
> Depois da correção, o cold start real é de **1,9%** (35.453 alunos, 51 municípios).
> Ver [`reports/03_features.md`](03_features.md).
>
> O texto original fica preservado porque a revisão faz parte do percurso analítico — e
> porque o erro só apareceu quando a interpretabilidade foi aplicada, o que é em si um
> argumento a favor dela.

**23,1% dos alunos de 2024** (428.119, em 676 municípios) estão em municípios sem histórico
em 2023. São municípios que entraram na avaliação agora — do ponto de vista de triagem, são
os que mais precisam de estimativa de risco, e é onde o modelo depende inteiramente do
contexto socioeconômico. Serão avaliados separadamente.

---

## Decisões que esta EDA sustenta

| # | Decisão | Evidência |
|---|---|---|
| 1 | Descartar `proficiencia` | `alfabetizado == (proficiencia ≥ 743)` em 100,0000% dos casos |
| 2 | `presenca`/`preenchimento` viram filtro | 100% dos ausentes têm `alfabetizado=0` por codificação administrativa |
| 3 | Sem tratamento de desbalanceamento | Alvo em 59/41 |
| 4 | **Nenhuma feature histórica por escola** | ID re-sorteado anualmente (2,4% de coincidência) |
| 5 | Features municipais como base | Código IBGE estável (r = 0,70) |
| 6 | Manter as 3 fontes de enriquecimento | Cobertura ≈100%, sinal moderado mas coerente |
| 7 | Baseline = taxa municipal de 2023 | AUC 0,63 com uma única variável |
| 8 | `GroupKFold` por município | Alunos do mesmo município compartilham todas as features |
| 9 | Flag + avaliação separada de cold start | 23,1% dos alunos ⚠️ revisado para 1,9% na Fase 5 — ver nota acima |
| 10 | Reportar F1 e AUC-PR além de acurácia | Custo assimétrico: não sinalizar criança em risco é pior |

## Limitações (vão para o README final)

- **Sem atributos individuais do aluno** — anonimização LGPD removeu sexo, raça/cor, idade e
  condição socioeconômica. Todo o sinal é contextual.
- **`id_escola` não rastreável** — impede o nível de análise mais informativo.
- **IDHM de 2010** — o Atlas só publica em anos censitários; 14 anos de defasagem,
  parcialmente compensados por PIB e população de 2023.
- **Apenas dois ciclos (2023, 2024)** — não há série longa para estimar tendência; só existe
  "o ano anterior".
