"""
Simplified evaluation without RAGAS - much faster!
Measures improvement directly
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import weaviate
from weaviate.classes.query import MetadataQuery
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Config
WEAVIATE_PORT = 8080
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "ProductionPapers"
MODEL_NAME = "llama3.2:3b"

# Load embedder for similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

TEST_QUESTIONS = [
    {
        "question": "What are the main risk factors for type 2 diabetes?",
        "ground_truth": "obesity physical inactivity family history age diet",
        "keywords": ["obesity", "diet", "physical activity", "genetics", "age"]
    },
    {
        "question": "How effective are COVID-19 vaccines?",
        "ground_truth": "vaccines highly effective preventing severe disease hospitalization",
        "keywords": ["vaccine", "efficacy", "prevention", "severe", "hospitalization"]
    },
    {
        "question": "What is cancer immunotherapy?",
        "ground_truth": "immunotherapy stimulates immune system attack cancer cells",
        "keywords": ["immune", "therapy", "checkpoint", "cells", "treatment"]
    },
]

def retrieve_documents(question: str, top_k: int = 5) -> List[Dict]:
    """Simple retrieval"""
    try:
        client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=50051, skip_init_checks=True)
        collection = client.collections.get(COLLECTION_NAME)
        
        response = collection.query.near_text(query=question, limit=top_k)
        
        contexts = []
        for obj in response.objects:
            contexts.append({
                "text": obj.properties.get("abstract", ""),
                "topic": obj.properties.get("topic", "")
            })
        
        client.close()
        return contexts
    except Exception as e:
        print(f"Error: {e}")
        return []

def generate_answer(question: str, contexts: List[Dict]) -> str:
    """Generate answer"""
    context_text = "\n\n".join([f"Doc {i+1}: {c['text'][:300]}" for i, c in enumerate(contexts)])
    
    prompt = f"""Answer briefly based on context.

Context:
{context_text}

Question: {question}

Brief answer (2-3 sentences):"""
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
    except:
        pass
    return "Could not generate answer"

def calculate_metrics(question_data: Dict, contexts: List[Dict], answer: str) -> Dict:
    """Simple metrics"""
    
    # 1. Context Precision: keyword overlap in retrieved docs
    keywords = question_data["keywords"]
    relevant_docs = sum(
        1 for ctx in contexts 
        if any(kw.lower() in ctx["text"].lower() for kw in keywords)
    )
    context_precision = relevant_docs / len(contexts) if contexts else 0
    
    # 2. Answer Relevancy: semantic similarity to ground truth
    if answer and question_data["ground_truth"]:
        answer_emb = embedder.encode([answer.lower()])
        gt_emb = embedder.encode([question_data["ground_truth"]])
        answer_relevancy = float(cosine_similarity(answer_emb, gt_emb)[0][0])
    else:
        answer_relevancy = 0
    
    # 3. Answer contains keywords?
    keyword_coverage = sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)
    
    return {
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy,
        "keyword_coverage": keyword_coverage
    }

def run_evaluation(name: str = "baseline") -> Dict:
    """Run evaluation"""
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} Evaluation")
    print(f"{'='*60}")
    
    results = []
    
    for i, item in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {item['question']}")
        
        # Retrieve
        contexts = retrieve_documents(item["question"])
        print(f"  Retrieved: {len(contexts)} docs")
        
        # Generate
        answer = generate_answer(item["question"], contexts)
        print(f"  Answer: {answer[:80]}...")
        
        # Evaluate
        metrics = calculate_metrics(item, contexts, answer)
        print(f"  Metrics: Precision={metrics['context_precision']:.2f}, Relevancy={metrics['answer_relevancy']:.2f}")
        
        results.append({
            "question": item["question"],
            "metrics": metrics
        })
        
        time.sleep(1)
    
    # Average metrics
    avg_metrics = {
        "context_precision": np.mean([r["metrics"]["context_precision"] for r in results]),
        "answer_relevancy": np.mean([r["metrics"]["answer_relevancy"] for r in results]),
        "keyword_coverage": np.mean([r["metrics"]["keyword_coverage"] for r in results]),
    }
    
    report = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": avg_metrics,
        "details": results
    }
    
    # Save
    Path(f"{name}_results.json").write_text(json.dumps(report, indent=2))
    
    # Print
    print(f"\n{'='*60}")
    print(f"{name.upper()} RESULTS")
    print(f"{'='*60}")
    for metric, score in avg_metrics.items():
        print(f"{metric:25s}: {score:.3f}")
    print(f"{'='*60}")
    
    return report

if __name__ == "__main__":
    baseline = run_evaluation("baseline")