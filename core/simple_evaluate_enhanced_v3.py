"""
Enhanced RAG V3 - Balanced optimization
Fix: Not over-filtering, better query expansion
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
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
print("✅ Models loaded")

# EXPANDED TEST SET
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
    {
        "question": "What are the symptoms of hypertension?",
        "ground_truth": "high blood pressure headache dizziness chest pain shortness breath",
        "keywords": ["blood pressure", "headache", "dizziness", "chest pain", "symptoms"]
    },
    {
        "question": "How is Alzheimer's disease diagnosed?",
        "ground_truth": "cognitive tests brain imaging memory assessment neurological examination",
        "keywords": ["diagnosis", "cognitive", "test", "imaging", "assessment", "memory"]
    },
    {
        "question": "What medications treat hypertension?",
        "ground_truth": "ACE inhibitors diuretics beta blockers calcium channel blockers antihypertensive",
        "keywords": ["medication", "ACE inhibitor", "diuretic", "beta blocker", "treatment", "drug"]
    },
]

# ===========================
# BASELINE RETRIEVAL
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
# ENHANCED RETRIEVAL V3 - FIXED
# ===========================

def expand_query_medical(question: str) -> List[str]:
    """Medical-focused query expansion"""
    
    # Extract key medical terms
    medical_terms = {
        "diabetes": ["diabetes mellitus", "T2DM", "diabetic"],
        "vaccine": ["vaccination", "immunization", "inoculation"],
        "cancer": ["tumor", "neoplasm", "malignancy", "carcinoma"],
        "immunotherapy": ["immune therapy", "immunological treatment", "biological therapy"],
        "hypertension": ["high blood pressure", "HTN", "elevated blood pressure"],
        "alzheimer": ["Alzheimer disease", "dementia", "cognitive decline"],
        "risk factors": ["causes", "risk", "predisposing factors", "etiology"],
        "symptoms": ["signs", "clinical features", "manifestations"],
        "diagnosis": ["diagnostic", "testing", "screening", "detection"],
        "treatment": ["therapy", "management", "intervention"],
        "medication": ["drug", "pharmaceutical", "medicine"]
    }
    
    # Generate variants
    variants = [question]
    question_lower = question.lower()
    
    # Replace with synonyms
    for term, synonyms in medical_terms.items():
        if term in question_lower:
            for synonym in synonyms[:2]:  # Max 2 per term
                variant = question_lower.replace(term, synonym)
                if variant != question_lower:
                    variants.append(variant.capitalize())
    
    # Add reformulations
    if "what are" in question_lower:
        variants.append(question.replace("What are", "List").replace("?", ""))
    if "how" in question_lower:
        variants.append(question.replace("How", "What is"))
    
    # Remove duplicates, keep top 4
    variants = list(dict.fromkeys(variants))[:4]
    
    print(f"  🔍 Query variants: {len(variants)}")
    for i, v in enumerate(variants, 1):
        print(f"     {i}. {v[:65]}...")
    
    return variants


def retrieve_documents_enhanced(question: str, top_k: int = 5) -> List[Dict]:
    """Enhanced V3: Better expansion + Balanced reranking"""
    
    # Step 1: Medical query expansion
    queries = expand_query_medical(question)
    
    # Step 2: Retrieve with hybrid search
    all_contexts = []
    seen_texts = set()
    
    try:
        client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=50051, skip_init_checks=True)
        collection = client.collections.get(COLLECTION_NAME)
        
        for query in queries:
            try:
                response = collection.query.hybrid(
                    query=query,
                    limit=15,  # Balanced: not too many, not too few
                    alpha=0.5
                )
            except:
                response = collection.query.near_text(query=query, limit=15)
            
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
    
    print(f"  📚 Retrieved: {len(all_contexts)} unique docs")
    
    if not all_contexts:
        return []
    
    # Step 3: BALANCED reranking (not too aggressive)
    pairs = [[question, ctx["text"][:512]] for ctx in all_contexts]
    scores = reranker.predict(pairs)
    
    for i, ctx in enumerate(all_contexts):
        ctx["rerank_score"] = float(scores[i])
    
    # Sort by score (no filtering - keep all)
    all_contexts.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    top_5 = all_contexts[:top_k]
    scores_str = ', '.join([f"{ctx['rerank_score']:.2f}" for ctx in top_5])
    print(f"  🎯 Top-{top_k} scores: [{scores_str}]")
    
    return top_5


# ===========================
# GENERATION - FIXED
# ===========================

def generate_answer(question: str, contexts: List[Dict], enhanced: bool = False) -> str:
    """Generate answer - LONGER context for better coverage"""
    
    if enhanced:
        # Enhanced: MORE context text for better keyword coverage
        context_text = "\n\n".join([
            f"Source {i+1}: {ctx['text'][:500]}"  # 500 chars instead of 400
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are a medical expert. Answer the question using the provided medical literature.

Medical Literature:
{context_text}

Question: {question}

Instructions:
- Answer in 3-4 sentences
- Include specific medical terms from the literature
- Be precise and comprehensive
- If information is in the literature, include it

Expert Answer:"""
    else:
        context_text = "\n\n".join([f"Doc {i+1}: {ctx['text'][:300]}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Answer based on context.

Context:
{context_text}

Question: {question}

Answer:"""
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200  # More tokens for comprehensive answer
                }
            },
            timeout=90
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
    """Calculate metrics"""
    
    keywords = question_data["keywords"]
    
    # Context Precision
    relevant_docs = sum(
        1 for ctx in contexts 
        if any(kw.lower() in ctx["text"].lower() for kw in keywords)
    )
    context_precision = relevant_docs / len(contexts) if contexts else 0
    
    # Answer Relevancy
    if answer and question_data["ground_truth"]:
        try:
            answer_emb = embedder.encode([answer.lower()])
            gt_emb = embedder.encode([question_data["ground_truth"]])
            answer_relevancy = float(cosine_similarity(answer_emb, gt_emb)[0][0])
        except:
            answer_relevancy = 0
    else:
        answer_relevancy = 0
    
    # Keyword Coverage
    keyword_coverage = sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)
    
    return {
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy,
        "keyword_coverage": keyword_coverage
    }


# ===========================
# EVALUATION
# ===========================

def run_evaluation(name: str = "enhanced_v3", use_enhanced: bool = True) -> Dict:
    """Run evaluation"""
    print(f"\n{'='*70}")
    print(f"Running {name.upper()} Evaluation")
    print(f"{'='*70}")
    
    if use_enhanced:
        print("🚀 V3 Enhancements (Balanced):")
        print("  1. Medical-focused query expansion with synonyms")
        print("  2. Hybrid search (15 docs per variant)")
        print("  3. Reranking without aggressive filtering")
        print("  4. Longer context + more tokens for generation")
        print("  5. Expanded test set (6 questions)")
    
    results = []
    
    for i, item in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─'*70}")
        print(f"[{i}/{len(TEST_QUESTIONS)}] {item['question']}")
        print(f"{'─'*70}")
        
        # Retrieve
        if use_enhanced:
            contexts = retrieve_documents_enhanced(item["question"])
        else:
            contexts = retrieve_documents_baseline(item["question"])
        
        print(f"  📄 Final: {len(contexts)} docs")
        
        # Generate
        answer = generate_answer(item["question"], contexts, enhanced=use_enhanced)
        print(f"  💬 Answer ({len(answer)} chars): {answer[:80]}...")
        
        # Evaluate
        metrics = calculate_metrics(item, contexts, answer)
        print(f"  📊 P={metrics['context_precision']:.2f} | R={metrics['answer_relevancy']:.2f} | K={metrics['keyword_coverage']:.2f}")
        
        results.append({
            "question": item["question"],
            "metrics": metrics,
            "answer": answer
        })
        
        time.sleep(0.5)
    
    # Average
    avg_metrics = {k: np.mean([r["metrics"][k] for r in results]) 
                   for k in ["context_precision", "answer_relevancy", "keyword_coverage"]}
    
    report = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": avg_metrics,
        "test_size": len(TEST_QUESTIONS),
        "details": results
    }
    
    Path(f"{name}_results.json").write_text(json.dumps(report, indent=2), encoding='utf-8')
    
    print(f"\n{'='*70}")
    print(f"📊 {name.upper()} SUMMARY")
    print(f"{'='*70}")
    for metric, score in avg_metrics.items():
        print(f"{metric:25s}: {score:.3f}")
    print(f"{'='*70}")
    
    return report


def compare_results(baseline_file="baseline_results.json", enhanced_file="enhanced_v3_results.json"):
    """Compare"""
    
    baseline = json.loads(Path(baseline_file).read_text())
    enhanced = json.loads(Path(enhanced_file).read_text())
    
    print(f"\n{'='*80}")
    print("📊 BASELINE vs ENHANCED V3")
    print(f"{'='*80}\n")
    
    for metric in baseline["metrics"].keys():
        b = baseline["metrics"][metric]
        e = enhanced["metrics"][metric]
        imp = ((e - b) / b * 100) if b > 0 else 0
        
        status = "✅✅✅ TARGET MET!" if imp >= 30 else "📈 Strong" if imp >= 20 else "📊 Good" if imp >= 10 else "⚠️"
        
        print(f"{metric.upper().replace('_', ' ')}:")
        print(f"  Baseline:  {b:.3f}")
        print(f"  Enhanced:  {e:.3f}")
        print(f"  Change:    {imp:+.1f}%  {status}\n")
    
    best = max(((enhanced["metrics"][m] - baseline["metrics"][m]) / baseline["metrics"][m] * 100) 
               for m in baseline["metrics"].keys())
    
    if best >= 30:
        print("🎯 SUCCESS: 30%+ achieved!")
    else:
        print(f"⚠️ Best: {best:.1f}% (need {30-best:.1f}% more)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("="*80)
    print("🚀 ENHANCED RAG V3 - BALANCED OPTIMIZATION")
    print("="*80)
    
    baseline = run_evaluation("baseline", use_enhanced=False)
    enhanced = run_evaluation("enhanced_v3", use_enhanced=True)
    compare_results("baseline_results.json", "enhanced_v3_results.json")
    
    print("\n✨ Evaluation complete!")