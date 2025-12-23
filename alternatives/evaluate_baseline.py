"""
Simplified RAG Evaluation Script with RAGAS
Uses smaller test set (30-50 questions) to work on 16GB RAM
"""

import json
import time
#from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pathlib import Path
from typing import List, Dict
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

# =========================
# CONFIG
# =========================

WEAVIATE_URL = "http://localhost:8080"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "ProductionPapers"
MODEL_NAME = "llama3.2:3b"
TOP_K = 5

# Test questions for medical RAG
TEST_QUESTIONS = [
    {
        "question": "What are the main risk factors for type 2 diabetes?",
        "ground_truth": "Main risk factors include obesity, sedentary lifestyle, family history, age over 45, and poor diet high in processed foods and sugar."
    },
    {
        "question": "How effective are COVID-19 vaccines in preventing severe disease?",
        "ground_truth": "COVID-19 vaccines are highly effective, showing 85-95% efficacy in preventing severe disease, hospitalization, and death across different variants."
    },
    {
        "question": "What is the mechanism of action for cancer immunotherapy?",
        "ground_truth": "Cancer immunotherapy works by stimulating or enhancing the immune system's ability to recognize and attack cancer cells, primarily through checkpoint inhibitors or CAR-T cell therapy."
    },
    {
        "question": "What lifestyle changes help manage hypertension?",
        "ground_truth": "Key lifestyle changes include reducing sodium intake, regular exercise, weight loss, limiting alcohol, stress management, and following the DASH diet."
    },
    {
        "question": "What are early warning signs of Alzheimer's disease?",
        "ground_truth": "Early signs include memory loss affecting daily activities, difficulty planning or solving problems, confusion with time or place, and changes in mood or personality."
    },
]

# =========================
# RAG PIPELINE
# =========================

def retrieve_documents(question: str, top_k: int = TOP_K) -> List[str]:
    """Retrieve relevant documents from Weaviate"""
    try:
        client = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,
            skip_init_checks=True
        )
        
        collection = client.collections.get(COLLECTION_NAME)
        
        response = collection.query.near_text(
            query=question,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        contexts = []
        for obj in response.objects:
            contexts.append(obj.properties.get("abstract", ""))
        
        client.close()
        return contexts
        
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []


def generate_answer(question: str, contexts: List[str]) -> str:
    """Generate answer using Ollama"""
    context_text = "\n\n".join([f"Document {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""You are a medical AI assistant. Answer the question using ONLY the provided context.

Context:
{context_text}

Question: {question}

Instructions:
- Be concise and directly address the question
- Use only information from the context
- If context doesn't answer the question, say "I cannot answer based on the provided context"

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
                    "top_p": 0.9,
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


def run_rag_pipeline(questions: List[Dict]) -> List[Dict]:
    """Run full RAG pipeline on test questions"""
    results = []
    
    for i, item in enumerate(questions, 1):
        print(f"\n📝 Processing question {i}/{len(questions)}")
        print(f"Q: {item['question']}")
        
        # Retrieve
        contexts = retrieve_documents(item["question"])
        print(f"   Retrieved {len(contexts)} documents")
        
        # Generate
        answer = generate_answer(item["question"], contexts)
        print(f"   Generated answer: {answer[:100]}...")
        
        results.append({
            "question": item["question"],
            "contexts": contexts,
            "answer": answer,
            "ground_truth": item["ground_truth"]
        })
        
        time.sleep(1)  # Rate limiting
    
    return results


# =========================
# EVALUATION
# =========================

def evaluate_rag(results: List[Dict]) -> Dict:
    """Evaluate RAG pipeline using RAGAS metrics"""
    
    # Convert to RAGAS dataset format
    data = {
        "question": [r["question"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "answer": [r["answer"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    }
    
    dataset = Dataset.from_dict(data)
    
    # Define metrics to evaluate
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
    
    print("\n🔬 Running RAGAS evaluation...")
    print("   This may take a few minutes...")
    
    # Evaluate
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=Ollama(model="llama3.2:3b", base_url="http://localhost:11434"),  # ← Добавить
        embeddings=OllamaEmbeddings(model="llama3.2:3b", base_url="http://localhost:11434"),  # ← Добавить
    )
    
    return results


# =========================
# REPORTING
# =========================

def generate_report(eval_results: Dict, output_file: str = "baseline_results.json"):
    """Generate evaluation report"""
    
    # FIX: eval_results может быть Dataset object, нужно преобразовать
    if hasattr(eval_results, 'to_pandas'):
        df = eval_results.to_pandas()
        metrics_dict = {
            col: float(df[col].mean()) 
            for col in df.columns 
            if col not in ['question', 'contexts', 'answer', 'ground_truth']
        }
    else:
        metrics_dict = {k: float(v) for k, v in eval_results.items()}
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_dict,
        "summary": {
            "avg_score": float(sum(metrics_dict.values()) / len(metrics_dict)),
            "test_size": len(TEST_QUESTIONS)
        }
    }


    
    # Save to file
    Path(output_file).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )
    
    # Print report
    print("\n" + "="*60)
    print("📊 EVALUATION REPORT")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Test Size: {report['summary']['test_size']} questions")
    print("\nMetrics:")
    print("-" * 60)
    
    for metric, score in report["metrics"].items():
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{metric:25s}: {score:.3f} {bar}")
    
    print("-" * 60)
    print(f"Average Score: {report['summary']['avg_score']:.3f}")
    print(f"\n✅ Report saved to: {output_file}")
    
    return report


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("🚀 Starting RAG Evaluation Pipeline")
    print("="*60)
    
    # Step 1: Run RAG pipeline
    print("\n📋 Step 1: Running RAG Pipeline")
    rag_results = run_rag_pipeline(TEST_QUESTIONS)
    
    # Step 2: Evaluate with RAGAS
    print("\n📋 Step 2: Evaluating with RAGAS")
    eval_results = evaluate_rag(rag_results)
    
    # Step 3: Generate report
    print("\n📋 Step 3: Generating Report")
    report = generate_report(eval_results)
    
    print("\n✨ Evaluation Complete!")
