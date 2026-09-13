# Roteiro do vídeo executivo — 5 minutos

**Formato pedido pelo enunciado:** reunião executiva com gestores públicos, lideranças ou
stakeholders. Quatro blocos obrigatórios: problema educacional · principais insights · valor
estratégico · como o modelo apoia políticas públicas.

**Princípio de corte:** cabe cerca de 750 palavras faladas em 5 minutos. Tudo que é rigor
metodológico — correção de ingestão, espaço de log-odds, seleção de hiperparâmetros — fica
**fora** e permanece disponível nos relatórios do repositório. Numa reunião executiva, o
gestor precisa saber o que decidir, não como o modelo foi construído.

**Tom:** você está apresentando para quem decide orçamento, não para quem revisa código.
Nenhuma sigla técnica sem tradução imediata.

---

## Bloco 1 — O problema (≈ 45s)

> Todo ano, cerca de 2 milhões de crianças brasileiras são avaliadas ao fim do 2º ano do
> ensino fundamental para saber se já estão alfabetizadas. Em 2024, **4 em cada 10 não
> estavam**.
>
> O Compromisso Nacional Criança Alfabetizada fixou metas para cada município até 2030. Mas
> há um problema de tempo: o gestor só descobre que um município ficou para trás **depois**
> da avaliação — quando aquele grupo de crianças já passou de ano.
>
> A pergunta que trouxemos para cá é simples: **dá para saber antes?**

**Apoio visual:** um único número grande na tela — 40% — e o mapa do Brasil.

---

## Bloco 2 — Principais insights (≈ 100s)

> Construímos um modelo que usa apenas informação disponível **antes** da avaliação: o
> desempenho do município no ano anterior, as metas pactuadas e o contexto socioeconômico.
> Três coisas ficaram claras.
>
> **Primeira: o resultado da criança é determinado muito mais pelo lugar do que por ela.**
> Entre os estados, a taxa de alfabetização vai de **36% a 85%** — quase 50 pontos de
> diferença. Uma criança no estado mais bem colocado tem mais que o dobro de chance de se
> alfabetizar que outra no pior. Isso não é sobre talento individual; é sobre condição de
> oferta.
>
> **Segunda: 38% dos municípios estão fora de rota para a própria meta de 2024** — quase 800
> mil crianças em redes que não vão cumprir o que pactuaram.
>
> **Terceira, e a mais útil: estar mal e estar fora da meta não são a mesma coisa.** O Sul
> tem o terceiro melhor desempenho do país e, ao mesmo tempo, a maior proporção de municípios
> fora de rota — dois terços deles. Não é contradição: as metas do Sul foram pactuadas num
> patamar mais ambicioso.
>
> São dois problemas diferentes, e hoje eles são tratados como um só. Um pede apoio
> pedagógico. O outro pede conversa sobre a meta.

**Apoio visual:** gráfico de barras por região comparando as duas colunas — risco
educacional × fração fora de rota. É o slide que sustenta a terceira afirmação.

---

## Bloco 3 — Valor estratégico (≈ 100s)

> O que o modelo entrega, na prática, é **uma fila de prioridade**.
>
> Testamos em municípios que o modelo nunca tinha visto. Separando-os em dez faixas de risco
> previsto: na faixa que o modelo apontou como mais crítica, a taxa real de alfabetização foi
> de **36%**. Na faixa que ele apontou como mais tranquila, **84%**. Quase 50 pontos de
> separação — o modelo acerta a ordem.
>
> E acerta melhor que a alternativa que existe hoje. Hoje, na prática, se usa o resultado do
> ano anterior para supor o próximo. O modelo é **significativamente mais preciso que isso**,
> porque combina o histórico com participação na avaliação, contexto socioeconômico e
> infraestrutura.
>
> Quero ser transparente sobre um limite, porque ele muda o uso correto da ferramenta. Os
> dados são anonimizados por lei e **não contêm nenhuma informação sobre a criança** — nem
> idade, nem condição social. O modelo não consegue, e não deve, apontar crianças
> individualmente.
>
> **O que ele faz bem é ordenar municípios.** E é assim que o dinheiro público é alocado de
> qualquer forma: por rede, não por aluno.

**Apoio visual:** `images/ranking_por_decil.png` — o gráfico das dez faixas. É a prova visual
mais forte que temos.

---

## Bloco 4 — Como apoia a política pública (≈ 85s)

> Três usos imediatos.
>
> **Primeiro: priorizar a agenda do ano.** A lista de municípios em risco sai antes da
> avaliação acontecer, com tempo hábil para formação de professores e apoio técnico chegarem
> enquanto ainda podem mudar o resultado daquela turma.
>
> **Segundo: calibrar pelo tamanho do programa.** O ponto de corte é ajustável. Se o programa
> tem capacidade para atender 30% da rede, o modelo entrega os 30% mais críticos. Se dobrar o
> orçamento, entrega mais. Isso é configuração, não um novo projeto.
>
> **Terceiro, e o mais acionável:** o agrupamento dos municípios por perfil revelou um bloco
> de **295 municípios** — pouco mais de 5% do país — que concentram os piores indicadores:
> quase o **triplo** da média nacional em saneamento inadequado e o **dobro** de crianças
> fora da escola. São também os de pior resultado educacional.
>
> Duzentos e noventa e cinco municípios é um alvo que cabe num programa. "O Nordeste vai mal"
> não é.
>
> E o perfil desse grupo sugere algo importante: a intervenção mais eficaz ali pode não ser
> primariamente pedagógica. Quando a criança não chega à escola e não tem água tratada em
> casa, o gargalo está antes da sala de aula.

**Apoio visual:** `images/cluster_municipios.png`, destacando o segmento crítico.

---

## Fechamento (≈ 20s)

> Deixo uma recomendação que não custa orçamento. Hoje o identificador das escolas na base
> pública é **trocado a cada ano**, o que impede acompanhar qualquer escola ao longo do
> tempo. Medimos que **17,7% da diferença de resultado acontece entre escolas do mesmo
> município** — é sinal real, e hoje ele é invisível.
>
> Publicar um identificador estável, sem identificar ninguém, abriria essa camada inteira de
> análise. É uma decisão de publicação de dado, não de investimento.
>
> Obrigado.

---

## O que ficou deliberadamente de fora

| Tema | Por que não entra |
|---|---|
| Correção do histórico de SP | Rigor de processo — importante para a banca ler, irrelevante para o gestor decidir |
| Armadilha do log-odds no SHAP | Detalhe de implementação |
| Troca do proxy pela variável real | Excelente achado técnico, sem consequência para a decisão |
| Hiperparâmetros, Optuna, folds | O gestor não precisa saber como o modelo foi ajustado |
| Números de AUC | Substituídos pelo que eles significam: "36% contra 84% entre as faixas extremas" |
| Suíte de 49 testes | Credibilidade se demonstra pelo resultado, não pelo processo, neste formato |

Tudo isso está documentado em `reports/01` a `reports/06` e no README.

---

## Checklist de gravação

- [ ] Cronometrar: 5 minutos é o **teto**, não a meta — mirar 4min30
- [ ] Nenhuma sigla sem tradução (AUC, SHAP, K-Means não devem ser ditos)
- [ ] Quatro blocos do enunciado visivelmente cobertos
- [ ] Postura de reunião: falar com quem decide, não com quem avalia o código
- [ ] Encerrar com a recomendação de dado — é o que diferencia de uma apresentação de modelo
