"""
Documents de démonstration : résumés de papers IA/ML fondateurs
Ces documents sont indexés au démarrage pour pouvoir tester le RAG immédiatement.
"""

from typing import List, Dict


def get_sample_documents() -> List[Dict]:
    """Retourne des résumés de papers IA/ML fondateurs pour la démo."""
    return [
        {
            "title": "Attention Is All You Need (Transformer)",
            "source": "Vaswani et al., 2017 — arXiv:1706.03762",
            "content": """
            The Transformer is a model architecture that relies entirely on self-attention mechanisms
            to compute representations of its input and output without using sequence-aligned RNNs or convolution.
            The model consists of an encoder and a decoder, each composed of a stack of identical layers.
            Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a 
            position-wise fully connected feed-forward network. Residual connections and layer 
            normalization are employed around each sub-layer.
            
            The attention function maps a query and a set of key-value pairs to an output.
            Multi-head attention allows the model to jointly attend to information from different 
            representation subspaces at different positions. The Transformer achieves state-of-the-art 
            performance on machine translation tasks. On the WMT 2014 English-to-German translation task,
            the big Transformer model outperforms all previously reported models including ensembles,
            establishing a new state-of-the-art BLEU score of 28.4.
            
            Positional encoding is added to the input embeddings to inject information about the 
            relative or absolute position of the tokens in the sequence. The Transformer's parallelizable
            architecture reduces training time significantly compared to recurrent models.
            Self-attention connects all positions with a constant number of operations, while recurrent
            layers require O(n) sequential operations.
            """
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "source": "Devlin et al., 2018 — arXiv:1810.04805",
            "content": """
            BERT (Bidirectional Encoder Representations from Transformers) is designed to pre-train 
            deep bidirectional representations from unlabeled text by jointly conditioning on both 
            left and right context in all layers. BERT uses two pre-training objectives:
            Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
            
            In MLM, 15% of input tokens are randomly masked and the model predicts them.
            NSP trains the model to understand relationships between sentences.
            BERT is pre-trained on BooksCorpus (800M words) and English Wikipedia (2500M words).
            
            Fine-tuning BERT achieves state-of-the-art results on eleven NLP benchmarks.
            On GLUE benchmark, BERT achieves 80.4% accuracy. On SQuAD v1.1, BERT achieves 
            93.2 F1 score, surpassing human performance of 91.2. BERT Base has 110M parameters
            and BERT Large has 340M parameters. The bidirectional nature of BERT is crucial:
            unlike GPT which uses left-to-right context, BERT uses full context from both directions.
            """
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "source": "Brown et al., 2020 — arXiv:2005.14165",
            "content": """
            GPT-3 is an autoregressive language model with 175 billion parameters, 10x more than 
            any previous non-sparse language model. GPT-3 is trained on 300 billion tokens from 
            Common Crawl, WebText2, Books1, Books2, and Wikipedia datasets.
            
            GPT-3 achieves strong performance on many NLP datasets in the few-shot setting,
            sometimes even matching state-of-the-art fine-tuned systems. Few-shot learning means
            the model is given only a few examples (typically 10-100) in the prompt, without 
            gradient updates. On SuperGLUE benchmark, GPT-3 achieves 71.8, approaching human 
            performance on some tasks.
            
            The model uses the same architecture as GPT-2 with modifications: alternating dense 
            and locally banded sparse attention patterns. In-context learning is a key capability:
            the model can perform tasks by analogy from examples in the prompt without fine-tuning.
            GPT-3 can generate human-quality text, write code, translate languages, and answer 
            questions across diverse domains.
            """
        },
        {
            "title": "Retrieval-Augmented Generation (RAG)",
            "source": "Lewis et al., 2020 — arXiv:2005.11401",
            "content": """
            Retrieval-Augmented Generation (RAG) combines parametric memory (LLM weights) with 
            non-parametric memory (external document retrieval) for knowledge-intensive NLP tasks.
            RAG models retrieve relevant documents using a dense retrieval component (DPR) and
            then use them to generate answers with a sequence-to-sequence model (BART).
            
            RAG addresses key limitations of pure LLMs: knowledge can be updated without 
            retraining, sources are transparent and verifiable, and hallucination is reduced.
            The retriever uses Maximum Inner Product Search (MIPS) over dense document embeddings.
            Documents are encoded offline, while queries are encoded at inference time.
            
            RAG outperforms parametric-only models on open-domain QA benchmarks:
            Natural Questions (44.5 EM), TriviaQA (56.8 EM), WebQuestions (45.5 EM).
            RAG generates more specific, diverse, and factual language than BART alone.
            The modular nature allows swapping retrievers and generators independently.
            """
        },
        {
            "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
            "source": "Reimers & Gurevych, 2019 — arXiv:1908.10084",
            "content": """
            Sentence-BERT (SBERT) modifies the BERT network using siamese and triplet network 
            structures to derive semantically meaningful sentence embeddings. SBERT is trained
            on NLI (Natural Language Inference) and STS (Semantic Textual Similarity) datasets.
            
            The key innovation: while BERT requires cross-encoding of sentence pairs (computationally
            expensive: O(n²) for n sentences), SBERT allows pre-computing sentence embeddings,
            reducing semantic search from 65 hours to 5 seconds for 10,000 sentences.
            
            SBERT uses mean pooling of BERT's token embeddings to create fixed-size sentence vectors.
            On STS benchmark, SBERT achieves Spearman correlation of 86.37, outperforming previous
            methods. For semantic search, cosine similarity between sentence vectors is highly effective.
            SBERT models like all-MiniLM-L6-v2 and all-mpnet-base-v2 are widely used in RAG systems
            as embedding models due to their speed-quality tradeoff.
            """
        },
    ]


def get_sample_qa_pairs() -> List[Dict]:
    """Dataset de référence pour l'évaluation — paires question/réponse attendue."""
    return [
        {
            "question": "What is the Transformer architecture and how does attention work?",
            "expected_answer": "The Transformer uses self-attention mechanisms without recurrence. "
                               "Multi-head attention allows attending to different representation subspaces. "
                               "It achieved 28.4 BLEU on WMT 2014 English-German translation."
        },
        {
            "question": "How does BERT differ from GPT in terms of training?",
            "expected_answer": "BERT is bidirectional using Masked Language Modeling and Next Sentence Prediction. "
                               "GPT uses left-to-right autoregressive generation. "
                               "BERT uses both left and right context while GPT only uses left context."
        },
        {
            "question": "What are the advantages of RAG over pure language models?",
            "expected_answer": "RAG allows knowledge updates without retraining, provides transparent sources, "
                               "and reduces hallucination by grounding answers in retrieved documents. "
                               "It combines parametric and non-parametric memory."
        },
        {
            "question": "How many parameters does GPT-3 have and what training data was used?",
            "expected_answer": "GPT-3 has 175 billion parameters and was trained on 300 billion tokens "
                               "from Common Crawl, WebText2, Books, and Wikipedia."
        },
        {
            "question": "What is Sentence-BERT and why is it useful for semantic search?",
            "expected_answer": "SBERT creates fixed-size sentence embeddings using siamese BERT networks. "
                               "It enables pre-computing embeddings for fast semantic search, "
                               "reducing search time from 65 hours to 5 seconds for 10,000 sentences."
        },
    ]
