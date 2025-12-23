"""
Enhanced RAG evaluation with improvements
Target: 30%+ improvement over baseline
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import weaviate
from weaviate.classes.query import MetadataQuery
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Config
WEAVIATE_PORT = 8080
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "ProductionPapers"
MODEL_NAME = "llama3.2:3b"

# Load models
print("📦 Loading models...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')  # Lightweight reranker
print("✅ Models loaded")

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

# ===========================
# BASELINE RETRIEVAL (simple)
# ===========================

def retrieve_documents_baseline(question: str, top_k: int = 5) -> List[Dict]:
    """Baseline: Simple vector search"""
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
        print(f"  ❌ Retrieval error: {e}")
        return []


# ===========================
# ENHANCED RETRIEVAL
# ===========================

def expand_query_simple(question: str) -> List[str]:
    """Simple query expansion with synonyms"""
    expansions = [question]
    
    # Add variations
    if "risk factors" in question.lower():
        expansions.append(question.replace("risk factors", "causes"))
        expansions.append(question.replace("What are", "List"))
    
    if "effective" in question.lower():
        expansions.append(question.replace("effective", "efficacy"))
        expansions.append(question.replace("How effective", "What is the effectiveness"))
    
    if "mechanism" in question.lower():
        expansions.append(question.replace("mechanism of action", "how does"))
        expansions.append(question.replace("What is", "Explain"))
    
    return expansions[:3]  # Max 3 variants


def retrieve_documents_enhanced(question: str, top_k: int = 5) -> List[Dict]:
    """Enhanced: Query expansion + Hybrid search + Reranking"""
    
    # Step 1: Query expansion
    queries = expand_query_simple(question)
    print(f"  🔍 Queries: {len(queries)} variants")
    
    # Step 2: Retrieve more documents (top-20)
    all_contexts = []
    seen_texts = set()
    
    try:
        client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=50051, skip_init_checks=True)
        collection = client.collections.get(COLLECTION_NAME)
        
        for query in queries:
            # Hybrid search (vector + BM25)
            try:
                response = collection.query.hybrid(
                    query=query,
                    limit=10,
                    alpha=0.5  # 50% vector, 50% BM25
                )
            except:
                # Fallback to vector search if hybrid not available
                response = collection.query.near_text(query=query, limit=10)
            
            for obj in response.objects:
                text = obj.properties.get("abstract", "")
                if text and text not in seen_texts:
                    all_contexts.append({
                        "text": text,
                        "topic": obj.properties.get("topic", "")
                    })
                    seen_texts.add(text)
        
        client.close()
        
    except Exception as e:
        print(f"  ❌ Retrieval error: {e}")
        return []
    
    print(f"  📚 Retrieved: {len(all_contexts)} docs")
    
    if not all_contexts:
        return []
    
    # Step 3: Rerank with cross-encoder
    pairs = [[question, ctx["text"][:512]] for ctx in all_contexts]  # Limit text length
    scores = reranker.predict(pairs)
    
    # Add scores and sort
    for i, ctx in enumerate(all_contexts):
        ctx["rerank_score"] = float(scores[i])
    
    all_contexts.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"  ✅ Reranked to top-{top_k} (best score: {all_contexts[0]['rerank_score']:.2f})")
    
    return all_contexts[:top_k]


# ===========================
# GENERATION
# ===========================

def generate_answer(question: str, contexts: List[Dict], enhanced: bool = False) -> str:
    """Generate answer"""
    
    if enhanced:
        # Enhanced prompt
        context_text = "\n\n".join([
            f"[Relevance: {ctx.get('rerank_score', 0):.2f}] {ctx['text'][:300]}"
            for ctx in contexts
        ])
        
        prompt = f"""You are a medical AI assistant. Answer concisely using ONLY the context.

Context:
{context_text}

Question: {question}

Instructions:
- Be concise (2-3 sentences)
- Focus on the specific question
- Use medical terminology appropriately
- If context insufficient, say "Cannot answer from context"

Concise answer:"""
    else:
        # Baseline prompt
        context_text = "\n\n".join([f"Doc {i+1}: {ctx['text'][:300]}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Answer briefly based on context.

Context:
{context_text}

Question: {question}

Brief answer (2-3 sentences):"""
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
    except Exception as e:
        print(f"  ❌ Generation error: {e}")
    
    return "Could not generate answer"


# ===========================
# METRICS
# ===========================

def calculate_metrics(question_data: Dict, contexts: List[Dict], answer: str) -> Dict:
    """Calculate evaluation metrics"""
    
    # 1. Context Precision: keyword overlap in retrieved docs
    keywords = question_data["keywords"]
    relevant_docs = sum(
        1 for ctx in contexts 
        if any(kw.lower() in ctx["text"].lower() for kw in keywords)
    )
    context_precision = relevant_docs / len(contexts) if contexts else 0
    
    # 2. Answer Relevancy: semantic similarity to ground truth
    if answer and question_data["ground_truth"]:
        try:
            answer_emb = embedder.encode([answer.lower()])
            gt_emb = embedder.encode([question_data["ground_truth"]])
            answer_relevancy = float(cosine_similarity(answer_emb, gt_emb)[0][0])
        except:
            answer_relevancy = 0
    else:
        answer_relevancy = 0
    
    # 3. Keyword coverage
    keyword_coverage = sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)
    
    return {
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy,
        "keyword_coverage": keyword_coverage
    }


# ===========================
# MAIN EVALUATION
# ===========================

def run_evaluation(name: str = "enhanced", use_enhanced: bool = True) -> Dict:
    """Run evaluation"""
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} Evaluation")
    print(f"{'='*60}")
    
    if use_enhanced:
        print("Enhancements: Query Expansion + Hybrid Search + Reranking")
    else:
        print("Baseline: Simple Vector Search")
    
    results = []
    
    for i, item in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {item['question']}")
        
        # Retrieve
        if use_enhanced:
            contexts = retrieve_documents_enhanced(item["question"])
        else:
            contexts = retrieve_documents_baseline(item["question"])
        
        print(f"  📄 Retrieved: {len(contexts)} docs")
        
        # Generate
        answer = generate_answer(item["question"], contexts, enhanced=use_enhanced)
        print(f"  💬 Answer: {answer[:80]}...")
        
        # Evaluate
        metrics = calculate_metrics(item, contexts, answer)
        print(f"  📊 Metrics: Precision={metrics['context_precision']:.2f}, Relevancy={metrics['answer_relevancy']:.2f}")
        
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
    Path(f"{name}_results.json").write_text(json.dumps(report, indent=2), encoding='utf-8')
    
    # Print
    print(f"\n{'='*60}")
    print(f"{name.upper()} RESULTS")
    print(f"{'='*60}")
    for metric, score in avg_metrics.items():
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"{metric:25s}: {score:.3f} {bar}")
    print(f"{'='*60}")
    
    return report


def compare_results(baseline_file: str = "baseline_results.json", 
                   enhanced_file: str = "enhanced_results.json"):
    """Compare baseline vs enhanced"""
    
    if not Path(baseline_file).exists():
        print(f"❌ {baseline_file} not found. Run baseline first!")
        return
    
    if not Path(enhanced_file).exists():
        print(f"❌ {enhanced_file} not found.")
        return
    
    baseline = json.loads(Path(baseline_file).read_text())
    enhanced = json.loads(Path(enhanced_file).read_text())
    
    print(f"\n{'='*80}")
    print("📊 BASELINE vs ENHANCED COMPARISON")
    print(f"{'='*80}\n")
    
    for metric in baseline["metrics"].keys():
        b_score = baseline["metrics"][metric]
        e_score = enhanced["metrics"][metric]
        improvement = ((e_score - b_score) / b_score * 100) if b_score > 0 else 0
        
        print(f"{metric.upper().replace('_', ' ')}:")
        print(f"  Baseline:  {b_score:.3f}")
        print(f"  Enhanced:  {e_score:.3f}")
        print(f"  Change:    {improvement:+.1f}%", end="")
        
        if improvement >= 30:
            print("  ✅ TARGET ACHIEVED (≥30%)")
        elif improvement >= 10:
            print("  📈 Good improvement")
        elif improvement >= 0:
            print("  📊 Minor improvement")
        else:
            print("  📉 Decreased")
        print()
    
    print(f"{'='*80}")


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    print("🚀 Enhanced RAG Evaluation")
    print("="*60)
    
    # Run enhanced
    enhanced = run_evaluation("enhanced", use_enhanced=True)
    
    # Compare with baseline
    print("\n" + "="*60)
    print("📊 Comparing with Baseline")
    print("="*60)
    compare_results()
    
    print("\n✨ Evaluation complete!")