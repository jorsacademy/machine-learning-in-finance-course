# Natural Language Processing in Finance

This section introduces practical NLP workflows for financial text while emphasizing timestamp integrity, licensing, source provenance, evaluation, and leakage control.

## Lessons

1. `01_financial_news_data_collection.py` — RSS-style collection, provenance, timestamps, and licensing considerations.
2. `02_text_preprocessing_financial_data.py` — conservative preprocessing that preserves finance-specific information such as numbers, negation, and tickers.
3. `03_sentiment_analysis_basics.py` — TF-IDF plus logistic regression as an interpretable sentiment baseline.
4. `04_named_entity_recognition.py` — transparent rule-based extraction of tickers, organizations, percentages, and monetary amounts.
5. `05_topic_modeling.py` — Latent Dirichlet Allocation for discovering recurring themes in a document collection.
6. `06_word_embeddings_financial_text.py` — dense semantic representations using TF-IDF and TruncatedSVD.
7. `07_bert_for_financial_documents.py` — optional finance-domain transformer inference with FinBERT.
8. `08_gpt_models_in_finance.py` — structured prompt design for extraction and summarization tasks without hard-coding an external API dependency.
9. `09_news_based_trading_signals.py` — timestamp-safe conversion of sentiment into a delayed trading signal with transaction costs.
10. `10_social_media_sentiment_analysis.py` — social sentiment aggregation with source-reliability weighting and manipulation warnings.

## Methodology notes

Financial NLP differs from generic NLP because seemingly small textual details can carry material meaning. Removing numbers, currency symbols, negation, ticker symbols, or domain terms can destroy information. Preprocessing should therefore be conservative and task-specific.

Publication time and data availability time are critical. A headline published after market close cannot be used to explain a decision that supposedly occurred before the headline existed. Historical datasets can also contain revised timestamps, duplicated articles, syndicated copies, or delayed ingestion. Store both source publication time and ingestion time whenever possible.

Sentiment is not the same as return prediction. A document can be positive about a company's operations while the information is already priced in, below expectations, or irrelevant to the chosen horizon. NLP outputs should be evaluated separately from any downstream trading strategy.

## Transformer and GPT-style models

Pretrained language models can improve classification, extraction, summarization, and semantic search, but they introduce additional concerns:

- model and tokenizer licensing;
- model versioning and reproducibility;
- hallucination risk in generative tasks;
- domain drift;
- context-window truncation;
- cost and latency;
- confidential-data handling;
- evaluation against labeled financial examples.

`07_bert_for_financial_documents.py` uses an optional pretrained model and therefore requires a separate model download. `08_gpt_models_in_finance.py` focuses on prompt structure and governance rather than tying the course to a rapidly changing vendor API.

## Social-media caution

Social platforms are especially vulnerable to coordinated promotion, bots, duplicate content, survivorship bias, deleted posts, and changing user populations. Raw post count or unweighted sentiment should not be treated as a reliable market signal without source-quality controls.

All examples are educational and do not constitute investment advice.
