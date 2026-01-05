# 📊 CrediTrust Complaint Intelligence System (RAG Foundation)

## 📌 Project Overview

CrediTrust Financial is a fast-growing digital finance company operating across East African markets with over 500,000 users. As the company scales, thousands of customer complaints are generated monthly across products such as Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. These complaints are rich in insights but are stored as unstructured text, making it difficult for internal teams to quickly identify emerging issues, risks, or product pain points.

This project lays the foundation for an internal AI-powered Retrieval-Augmented Generation (RAG) chatbot that transforms raw complaint narratives into a searchable, evidence-backed knowledge system. The goal is to reduce the time required to identify complaint trends from days to minutes, empower non-technical teams to access insights independently, and enable proactive, data-driven decision-making.

---

## 🎯 Business Objective

The primary objective is to convert large-scale unstructured complaint data into a strategic asset that supports Product, Support, Compliance, and Risk teams. By enabling semantic search and question-answering over customer complaints, internal stakeholders can quickly surface recurring issues, validate decisions with real customer evidence, and respond proactively to emerging problems.

---

## 🧩 Task 1: Exploratory Data Analysis & Text Preprocessing

The first phase focuses on understanding the structure, quality, and limitations of the raw complaint dataset, which contains over 9 million records and significant missing values. Exploratory analysis was conducted to examine complaint distribution across products, narrative length variability, and class imbalance. Based on this analysis, products were consolidated into four core categories aligned with business priorities.

Text preprocessing was then applied to clean and normalize complaint narratives by removing noise such as emails, phone numbers, and unnecessary symbols while preserving semantically important elements like negations. Custom stopword handling and normalization techniques were used to ensure the cleaned text remains suitable for downstream semantic retrieval. The result is a high-quality, cleaned dataset ready for embedding and retrieval tasks.

---

## 🧠 Task 2: Text Chunking, Embedding & Vector Store Indexing

The second phase prepares the cleaned complaint narratives for efficient semantic search. A stratified sampling strategy was used to select 10,000–15,000 complaints while preserving the original product distribution. Since long narratives are ineffective when embedded as a single vector, a chunking strategy was implemented and empirically evaluated using multiple chunk sizes and overlaps.

Based on qualitative and quantitative analysis, a chunk size of 500 characters with 100-character overlap was selected as the best balance between semantic coherence and retrieval granularity. Each text chunk is then embedded using the lightweight and high-performing sentence-transformers/all-MiniLM-L6-v2 model. The resulting embeddings, along with essential metadata such as complaint ID and product category, are stored in a persistent ChromaDB vector store, forming the backbone of the RAG system.