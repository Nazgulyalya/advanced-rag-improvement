"""
Enhanced RAG with Query Expansion and Reranking
Target: 30%+ improvement in Context Precision and Answer Relevancy
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import weaviate
from weaviate.classes.query import MetadataQuery
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from sentence_transformers import CrossEncoder

# =========================
# CONFIG
# =========================

WEAVIATE_URL = "http://localhost:8080"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "ProductionPapers"
MODEL_NAME = "llama3.2"
TOP_K = 5
RERANK_TOP_K = 20  # Retrieve more, then rerank

# Initialize cross-encoder for reranking (lightweight model)
# This will be loaded on first use
CROSS_ENCODER = None


# =========================
# ENHANCEMENT 1: QUERY EXPANSION
# =========================

def expand_query(question: str) -> List[str]:
    """
    Generate query variations using LLM
    Target: Better retrieval recall
    """
    prompt = f"""Generate 2 alternative phrasings of this medical question. 
Keep the meaning the same but use different words.
Output ONLY the 2 alternative questions, one per line, no numbering.

Original question: {question}

Alternative questions:"""
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            alternatives = response.json()["response"].strip().split("\n")
            # Clean up and filter
            alternatives = [alt.strip() for alt in alternatives if alt.strip()]
            alternatives = [alt for alt in alternatives if len(alt) > 10][:2]
            return [question] + alternatives
        
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
    
    return [question]  # Fallback to original


# =========================
# ENHANCEMENT 2: HYBRID SEARCH + RERANKING
# =========================

def retrieve_with_reranking(question: str, top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    """
    Enhanced retrieval with:
    1. Query expansion
    2. Hybrid search (vector + BM25)
    3. Cross-encoder reranking
    
    Target: 30-50% improvement in Context Precision
    """
    global CROSS_ENCODER
    
    # Initialize cross-encoder on first use
    if CROSS_ENCODER is None:
        print("📦 Loading cross-encoder model...")
        CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    try:
        client = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,
            skip_init_checks=True
        )
        
        collection = client.collections.get(COLLECTION_NAME)
        
        # Step 1: Query expansion
        queries = expand_query(question)
        print(f"   🔍 Expanded to {len(queries)} queries")
        
        # Step 2: Hybrid search for each query variant
        all_results = []
        seen_abstracts = set()
        
        for query in queries:
            response = collection.query.hybrid(
                query=query,
                limit=RERANK_TOP_K // len(queries) + 5,
                alpha=0.5  # 0.5 = balanced vector + BM25
            )
            
            for obj in response.objects:
                abstract = obj.properties.get("abstract", "")
                if abstract not in seen_abstracts and abstract:
                    all_results.append({
                        "text": abstract,
                        "score": obj.metadata.score if hasattr(obj.metadata, 'score') else 0.5
                    })
                    seen_abstracts.add(abstract)
        
        client.close()
        
        print(f"   📚 Retrieved {len(all_results)} unique documents")
        
        if not all_results:
            return [], []
        
        # Step 3: Rerank with cross-encoder
        pairs = [[question, doc["text"]] for doc in all_results]
        rerank_scores = CROSS_ENCODER.predict(pairs)
        
        # Combine and sort
        for i, doc in enumerate(all_results):
            doc["rerank_score"] = float(rerank_scores[i])
        
        all_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-k after reranking
        top_contexts = [doc["text"] for doc in all_results[:top_k]]
        top_scores = [doc["rerank_score"] for doc in all_results[:top_k]]
        
        print(f"   ✅ Reranked to top-{top_k} (scores: {top_scores[0]:.3f} to {top_scores[-1]:.3f})")
        
        return top_contexts, top_scores
        
    except Exception as e:
        print(f"❌ Enhanced retrieval error: {e}")
        return [], []


# =========================
# ENHANCEMENT 3: IMPROVED PROMPT
# =========================

ENHANCED_PROMPT_TEMPLATE = """You are a medical AI assistant specialized in evidence-based medicine.

**Task**: Answer the question concisely and accurately using ONLY the information from the provided medical abstracts.

**Medical Context**:
{context}

**Question**: {question}

**Instructions**:
1. Directly address the specific question asked
2. Be concise - provide a 2-3 sentence answer maximum
3. Use medical terminology appropriately
4. Focus only on information relevant to the question
5. If the context doesn't contain enough information to answer, respond EXACTLY with: "Based on the provided abstracts, I cannot fully answer this question."

**Concise Evidence-Based Answer**:"""


def generate_enhanced_answer(question: str, contexts: List[str], scores: List[float] = None) -> str:
    """
    Generate answer with improved prompt engineering
    Target: 20-40% improvement in Answer Relevancy
    """
    # Format contexts with quality indicators
    if scores:
        context_text = "\n\n".join([
            f"[Relevance: {score:.2f}] {ctx[:500]}..."  # Show relevance score
            for score, ctx in zip(scores, contexts)
        ])
    else:
        context_text = "\n\n".join([f"Abstract {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = ENHANCED_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return "Error generating answer"


# =========================
# ENHANCED RAG PIPELINE
# =========================

def run_enhanced_rag_pipeline(questions: List[Dict]) -> List[Dict]:
    """Run enhanced RAG pipeline with all improvements"""
    results = []
    
    for i, item in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 Question {i}/{len(questions)}")
        print(f"Q: {item['question']}")
        print(f"{'='*60}")
        
        # Enhanced Retrieval
        contexts, scores = retrieve_with_reranking(item["question"])
        
        if not contexts:
            print("   ⚠️ No documents retrieved")
            results.append({
                "question": item["question"],
                "contexts": [""],
                "answer": "No relevant context found",
                "ground_truth": item["ground_truth"]
            })
            continue
        
        # Enhanced Generation
        answer = generate_enhanced_answer(item["question"], contexts, scores)
        print(f"   💬 Answer: {answer[:150]}...")
        
        results.append({
            "question": item["question"],
            "contexts": contexts,
            "answer": answer,
            "ground_truth": item["ground_truth"]
        })
        
        time.sleep(1)
    
    return results


# =========================
# EVALUATION
# =========================

def evaluate_enhanced_rag(results: List[Dict]) -> Dict:
    """Evaluate enhanced RAG pipeline"""
    
    data = {
        "question": [r["question"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "answer": [r["answer"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    }
    
    dataset = Dataset.from_dict(data)
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
    
    print("\n🔬 Running RAGAS evaluation on enhanced system...")
    
    eval_results = evaluate(
        dataset,
        metrics=metrics,
    )
    
    return eval_results


# =========================
# COMPARISON & REPORTING
# =========================

def compare_results(baseline_file: str, enhanced_file: str):
    """Compare baseline vs enhanced results"""
    
    baseline = json.loads(Path(baseline_file).read_text())
    enhanced = json.loads(Path(enhanced_file).read_text())
    
    print("\n" + "="*80)
    print("📊 BASELINE vs ENHANCED COMPARISON")
    print("="*80)
    
    improvements = {}
    
    for metric in baseline["metrics"].keys():
        baseline_score = baseline["metrics"][metric]
        enhanced_score = enhanced["metrics"][metric]
        improvement = ((enhanced_score - baseline_score) / baseline_score) * 100
        improvements[metric] = improvement
        
        print(f"\n{metric.upper()}:")
        print(f"  Baseline:  {baseline_score:.3f}")
        print(f"  Enhanced:  {enhanced_score:.3f}")
        print(f"  Change:    {improvement:+.1f}%")
        
        if improvement >= 30:
            print(f"  ✅ TARGET ACHIEVED (≥30%)")
        elif improvement >= 0:
            print(f"  📈 Improved")
        else:
            print(f"  📉 Decreased")
    
    print("\n" + "="*80)
    
    # Check if target met
    target_met = any(imp >= 30 for imp in improvements.values())
    
    if target_met:
        print("🎯 SUCCESS: At least one metric improved by ≥30%!")
    else:
        print("⚠️ Target not met: No metric improved by 30%+")
    
    print("="*80)
    
    return improvements


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    from evaluate_baseline import TEST_QUESTIONS, generate_report
    
    print("🚀 Starting ENHANCED RAG Evaluation Pipeline")
    print("="*80)
    print("\nEnhancements:")
    print("  1. ✨ Query Expansion (2-3 variants per query)")
    print("  2. ✨ Hybrid Search (Vector + BM25)")
    print("  3. ✨ Cross-Encoder Reranking")
    print("  4. ✨ Improved Prompt Engineering")
    print("="*80)
    
    # Run enhanced pipeline
    print("\n📋 Step 1: Running Enhanced RAG Pipeline")
    rag_results = run_enhanced_rag_pipeline(TEST_QUESTIONS)
    
    # Evaluate
    print("\n📋 Step 2: Evaluating with RAGAS")
    eval_results = evaluate_enhanced_rag(rag_results)
    
    # Generate report
    print("\n📋 Step 3: Generating Report")
    report = generate_report(eval_results, "enhanced_results.json")
    
    # Compare with baseline
    print("\n📋 Step 4: Comparing with Baseline")
    if Path("baseline_results.json").exists():
        improvements = compare_results("baseline_results.json", "enhanced_results.json")
    else:
        print("⚠️ baseline_results.json not found. Run evaluate_baseline.py first!")
    
    print("\n✨ Enhanced Evaluation Complete!")
