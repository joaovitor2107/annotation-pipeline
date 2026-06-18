# Pipeline de Anotação de Stance para Tweets em Português

Pipeline semissupervisionado para classificar a posição (_stance_) de tweets em três categorias, expandindo um conjunto pequeno de anotações manuais para centenas de milhares de pseudo-labels usando BERT fine-tuned.

Inclui opcionalmente um pré-filtro de ironia para melhorar a qualidade dos pseudo-labels.

---

## Visão Geral

```
Anotações manuais (~3k)
        │
        ▼
  data_selection.py          → curated_tweets_stance.csv
        │
        ▼
  bert_annotator.py (train)  → models/bert_stance/
        │
        ▼
  bert_annotator.py (eval)   → Relatório de classificação no test set
        │
        ▼                    (opcional)
  bert_irony.py (train)      → models/bert_irony/
        │
        ▼
  bert_annotator.py (annotate, com --irony-filter)
        │                    → pseudo_labeled_stance.csv (~1M tweets)
        ▼                    (opcional, recomendado)
  threshold_calibration.py   → threshold_calibration.json (thresholds por classe)
        │
        ▼
  confidence_filter.py       → pseudo_labeled_filtered_perclass.csv (~230k tweets)
```

---

## Formato dos Dados

### 1. Anotações manuais (`data/anotacoes.csv`)

CSV com pelo menos duas colunas. O separador pode ser `;` ou `,`:

| conversation_id | Posicao_Final |
|---|---|
| 1234567890 | in favor |
| 9876543210 | against |
| 1122334455 | neutral |

Os valores aceitos para `Posicao_Final` são: `in favor`, `against`, `neutral`.

### 2. Tweets raw (`data/tweets/*.csv`)

Um ou mais CSVs com os tweets não anotados. Colunas mínimas:

| conversation_id | text |
|---|---|
| 1234567890 | "texto do tweet..." |

### 3. Dataset de treino (gerado automaticamente por `data_selection.py`)

`data/curated_tweets_stance.csv` — resultado da seleção, com colunas:
`conversation_id, text, label, split, [irony]`

A coluna `irony` (`ironic`/`not_ironic`) é opcional e só é usada pelo `bert_irony.py`.

---

## Instalação

```bash
pip install -r requirements.txt
```

Todos os scripts são executados diretamente na pasta `annotation-pipeline/`:

```bash
cd annotation-pipeline/
python bert_annotator.py --step train ...
```

---

## Passo a Passo

### Passo 0 — Preparar o conjunto de treino

Lê as anotações manuais, filtra conversas sem ambiguidade, balanceia por classe e gera o split train/val/test.

```bash
python data_selection.py \
    --annotations data/anotacoes.csv \
    --tweets-dir  data/tweets/ \
    --output      data/curated_tweets_stance.csv \
    --max-per-class 500
```

**Parâmetros principais:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--max-per-class` | 500 | Máximo de exemplos por classe |
| `--oversample` | off | Oversampling da classe minoritária no treino |

---

### Passo 1 (Opcional) — Treinar o detector de ironia

Só necessário se quiser usar o filtro `--irony-filter` na anotação.
Requer uma coluna `irony` no CSV de treino com valores `ironic`/`not_ironic`.

```bash
python bert_irony.py --step train \
    --freeze-layers 4 \
    --model-dir models/bert_irony
```

Avaliar:

```bash
python bert_irony.py --step eval --model-dir models/bert_irony
```

**Parâmetros principais:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--freeze-layers` | 4 | Número de camadas a descongelar (ver decisões abaixo) |
| `--irony-weight-multiplier` | 1.0 | Peso extra para a classe minoritária irônica |

---

### Passo 2 — Treinar o classificador de stance

```bash
python bert_annotator.py --step train \
    --freeze-layers 4 \
    --model-dir models/bert_stance
```

Avaliar no test set:

```bash
python bert_annotator.py --step eval \
    --model-dir models/bert_stance
```

**Parâmetros principais:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--freeze-layers` | 0 | 0 = full fine-tuning; 4 = congela 8/12 camadas |
| `--model-name` | BERTimbau | Qualquer modelo BERT do HuggingFace |
| `--scheduler` | linear | `linear` ou `cosine` (decay após warmup) |

---

### Passo 3 — Anotar o corpus completo

Sem filtro de ironia:

```bash
python bert_annotator.py --step annotate \
    --model-dir models/bert_stance \
    --tweets-dir data/tweets/ \
    --annotation-output data/pseudo_labeled_stance.csv
```

Com filtro de ironia (recomendado se o Passo 1 foi feito):

```bash
python bert_annotator.py --step annotate \
    --model-dir     models/bert_stance \
    --tweets-dir    data/tweets/ \
    --irony-filter \
    --irony-model-dir models/bert_irony \
    --irony-threshold 0.10 \
    --annotation-output data/pseudo_labeled_stance_noirony.csv
```

O processo é **retomável**: se interrompido, relê o arquivo de saída existente e pula os IDs já anotados.

---

### Passo 4 (Opcional, recomendado) — Calibrar thresholds por classe

Em vez de um threshold uniforme, escolhe um threshold por classe via calibração de precisão no test split humano: para cada classe, o menor threshold da grade `{0.90, 0.95, 0.97, 0.99, 0.995}` que atinge a precisão alvo (default: 0.90).

```bash
python threshold_calibration.py \
    --model-dir  models/bert_stance \
    --curated-csv data/curated_tweets_stance.csv \
    --pseudo-csv data/pseudo_labeled_stance_noirony.csv \
    --target-precision 0.90
```

O script:
- Roda o modelo treinado no test split (restrito a tweets não-irônicos por padrão, para coincidir com a anotação `--irony-filter`; use `--keep-ironic` para manter todos);
- Imprime a tabela threshold × precisão por classe;
- Projeta a distribuição do pool de pseudo-labels após o filtro;
- Salva tudo em `results/threshold_calibration.json` e imprime o comando pronto para o Passo 5.

**Parâmetros principais:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--target-precision` | 0.90 | Precisão mínima por classe no test split |
| `--keep-ironic` | off | Mantém tweets irônicos no split de calibração |

---

### Passo 5 — Filtrar por confiança

Retém apenas os tweets onde o modelo teve alta confiança na sua predição.

Com threshold uniforme:

```bash
python confidence_filter.py \
    --input     data/pseudo_labeled_stance_noirony.csv \
    --output    data/pseudo_labeled_filtered_0.97.csv \
    --threshold 0.97
```

Com thresholds por classe (saída do Passo 4; valores de exemplo do nosso corpus):

```bash
python confidence_filter.py \
    --input  data/pseudo_labeled_stance_noirony.csv \
    --output data/pseudo_labeled_filtered_perclass.csv \
    --threshold-per-class "a favor=0.99,contra=0.97,neutro=0.995"
```

Classes não listadas em `--threshold-per-class` usam o valor de `--threshold`.

---

## Decisões de Design

### Por que BERTimbau?

O [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (`neuralmind/bert-base-portuguese-cased`) é o modelo BERT pré-treinado em português mais estabelecido. Em um contexto político brasileiro com gírias e referências culturais específicas, um modelo pré-treinado em PT-BR supera modelos multilinguais genéricos.

Qualquer modelo BERT do HuggingFace pode ser usado via `--model-name`.

---

### Por que `--freeze-layers 4`?

O BERTimbau-base tem 12 camadas. Com `--freeze-layers 4`:
- As 8 camadas inferiores ficam congeladas (mantêm o conhecimento linguístico pré-treinado).
- As 4 camadas superiores + pooler + classificador são treinadas (adaptação ao domínio).
- Isso corresponde a ~26% dos parâmetros treináveis vs 100% no full fine-tuning.

**Por que não treinar tudo?** Com ~5000 exemplos de treino, o full fine-tuning tende a sofrer de _catastrophic forgetting_: o modelo "esquece" o português e se ajusta demais ao conjunto pequeno (overfitting). Congelar as camadas base funciona como regularização implícita.

Em comparação, `freeze-layers=4` deu F1-macro=0.697 vs 0.667 do full fine-tuning.

---

### Por que filtrar ironia?

Tweets irônicos são classificados com o _label_ errado pelo modelo de stance porque o texto aparenta uma posição que é o oposto do que o autor pretende. Por exemplo, _"Que povo democrático! 🤡"_ parece `neutro` mas é `contra`.

A estratégia escolhida é **ignorar** tweets irônicos (não incluí-los nos pseudo-labels), e não tentar inverter o label — porque o detector de ironia tem muitos falsos positivos, e inverter labels corretos piora o conjunto.

Com `--irony-threshold 0.10`, o detector usa um limiar baixo (alta sensibilidade / recall), privilegiando a remoção de irônicos reais mesmo à custa de também remover alguns não-irônicos. No nosso corpus, removeu ~18% dos tweets.

---

### Por que threshold de confiança 0.97?

A "confiança" aqui é o valor mais alto do softmax dos logits — _não_ uma probabilidade calibrada. Para um classificador com 3 classes, o máximo aleatório seria 0.33. Um valor ≥ 0.97 significa que o modelo atribui 97% da sua "massa" a uma só classe, sendo muito improvável que erre.

O trade-off é quantidade vs. qualidade:
- Threshold 0.85 → retém ~84% dos tweets, mais ruído
- Threshold 0.97 → retém ~42% dos tweets, menos ruído

Para treinar GNNs com labels semi-supervisionados, preferimos menos exemplos com sinal mais limpo.

---

### Por que thresholds por classe?

A confiança do softmax não é igualmente confiável em todas as classes. No nosso corpus, a precisão em t=0.90 era muito desigual: `a favor` e `contra` já eram limpas, mas `neutro` tinha precisão de apenas **0.65** — um threshold uniforme mantém muitos `neutro` errados e, ao aumentá-lo para compensar, descarta `a favor` corretos (a classe minoritária).

A solução é calibrar um threshold por classe no test split humano (Passo 4), exigindo precisão ≥ 0.90 em cada classe. No nosso caso isso resultou em:

| Classe | Threshold | Precisão no test split |
|---|---|---|
| a favor | 0.99 | 1.000 |
| contra | 0.97 | 0.910 |
| neutro | 0.995 | 1.000 |

O efeito é um pool de pseudo-labels menor (~23% de retenção vs ~42% com threshold uniforme 0.97) mas com distribuição de classes muito mais saudável (`neutro` cai de ~48% para ~24% do pool). Na nossa aplicação downstream (detecção de comunidades com GNNs), isso se traduziu em +4.7pp de pureza de stance (p<1e-10).

**Atenção:** esses thresholds são específicos do par modelo+corpus. Após retreinar o modelo ou mudar de corpus, recalibre com `threshold_calibration.py`.

---

### Por que inferência sempre em CPU (Apple Silicon)?

`BertForSequenceClassification` tem um bug numérico no backend MPS do PyTorch (Apple Silicon): o `argmax` dos logits produz resultados incorretos durante a inferência (F1 cai de ~0.70 para ~0.31). O treino em MPS funciona corretamente. A função `_get_device(infer=True)` sempre retorna `cpu` para inferência como workaround.

---

## Estrutura de Arquivos

```
annotation-pipeline/
├── README.md               # esta documentação
├── requirements.txt
├── data_selection.py       # Passo 0: seleção e preparação do conjunto de treino
├── bert_irony.py           # Passo 1 (opcional): detector de ironia
├── bert_annotator.py       # Passos 2-3: stance classifier + anotação
├── threshold_calibration.py # Passo 4 (opcional): calibração de thresholds por classe
├── confidence_filter.py    # Passo 5: filtro por confiança
└── tweet_stream.py         # utilitário: leitura lazy de CSVs de tweets
```
