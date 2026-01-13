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
---

## 🧩 Task 3: RAG System Development

In Task 3, a complete RAG pipeline was developed, consisting of the following core components:

### 🔹 Vector Store (FAISS)

Complaint documents are converted into dense embeddings.

FAISS is used as the vector database for efficient similarity search.

Given a user query, the system retrieves the top-k most relevant document chunks using cosine similarity.

### 🔹 Retriever

The retriever queries the FAISS index with the embedded user question.

It returns the most relevant complaint chunks to be used as context for generation.

The number of retrieved chunks is kept small to ensure memory efficiency and avoid context overflow.

### 🔹 Generator (LLM)
The generator uses a quantized Mistral-7B-Instruct model:

Model: mistral-7b-instruct.Q4_K_M.gguf

Format: GGUF

Runtime: CPU-only

The model is loaded locally using a lightweight inference backend.

The generator combines:

1. A system prompt (instructions)

2. Retrieved complaint chunks (context)

3. The user’s question

The LLM generates an answer based only on the retrieved context, reducing hallucinations.

This design enables efficient and reliable question answering over large complaint datasets without requiring a GPU.
---

## 🖥️ Task 4: Gradio-Based Chatbot Interface
In Task 4, a Gradio web interface was developed to make the RAG system interactive and user-friendly.

### 🔹 Features

Simple chat-style interface for entering user questions

Real-time response generation using the RAG pipeline

Fully local execution (no external APIs required)

Designed to work smoothly on low-resource machines

### 🔹 Workflow

User enters a question in the Gradio interface

The retriever fetches relevant complaint chunks from FAISS

The generator (Mistral-7B-Instruct, quantized) produces an answer

The response is displayed in the chatbot UI