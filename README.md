
# O Custo da Performance: Modelos de Visão e Linguagem para Estimativa de Idade sob Oclusão Facial (Paper ID 63)

Repositório destinado ao armazenamento de experimentos, códigos, prompts, métricas e análises relacionados à estimativa de idade utilizando modelos Vision-Language (VLMs) em tarefas de Visual Question Answering (VQA).Repositório destinado ao armazenamento de experimentos, códigos, prompts, métricas e análises relacionados à estimativa de idade utilizando modelos Vision-Language (VLMs) em tarefas de Visual Question Answering (VQA), desenvolvido como parte de um trabalho submetido para a 2ª Edição da Conferência Latino-Americana de Ética em Inteligência Artificial.

---

## Objetivo

Investigar a capacidade de modelos multimodais em estimar idade a partir de imagens, utilizando não apenas informações faciais, mas também pistas contextuais da cena.

O projeto avalia:
- diferentes estratégias de prompting;
- desempenho de modelos VLMs;
- impacto da oclusão facial;
- custo computacional e emissão de CO₂.

---

# Estrutura do Repositório

```text
estimativa-idade-vlm/
│
├── prompts/           # Prompts utilizados nos experimentos
└── scripts/           # Scripts de inferência, avaliação e visualização
```

---

# Pipeline Experimental

```text
Imagem → Prompt → Modelo VLM → Resposta textual → Pós-processamento → Avaliação
```

Os modelos recebem imagens e perguntas relacionadas à idade, produzindo respostas textuais posteriormente estruturadas para análise.

---

# Prompts Avaliados

O projeto avalia diferentes níveis de granularidade:
- classificação binária;
- faixa etária;
- estimativa exata da idade.

---

# Principais Achados

- Prompts mais objetivos apresentaram melhor desempenho.
- VLMs obtiveram resultados superiores em cenários com oclusão facial.
- Métodos tradicionais e VLMs tiveram desempenho semelhante com faces visíveis.
- A baixa representatividade de idosos impactou a estimativa exata de idade.
- Modelos multimodais apresentaram maior custo computacional.

---

