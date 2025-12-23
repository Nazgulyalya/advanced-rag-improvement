import requests
import time
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.config import Configure, Property, DataType
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

TOPICS = {
    "diabetes": "diabetes randomized controlled trial",
    "covid": "COVID vaccine randomized trial",
    "cancer": "cancer immunotherapy randomized",
    "hypertension": "hypertension randomized controlled",
    "alzheimer": "Alzheimer disease randomized trial"
}

ARTICLES_PER_TOPIC = 40
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "ProductionPapers"

# =========================
# FUNCTIONS
# =========================

def fetch_pmids(query, retmax):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    return [id_el.text for id_el in root.findall(".//Id")]


def fetch_abstracts(pmids, topic):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//AbstractText")

        # 🔴 HARD FILTER: ABSTRACT IS REQUIRED
        if not abstract_parts:
            continue

        abstract = " ".join(
            part.text.strip()
            for part in abstract_parts
            if part.text
        )

        if not abstract.strip():
            continue

        papers.append({
            "pmid": pmid_el.text if pmid_el is not None else "",
            "title": title_el.text.strip()[:200] if title_el is not None else "",
            "abstract": abstract[:1000],
            "topic": topic
        })

    return papers


# =========================
# MAIN PIPELINE
# =========================

all_papers = []

print("🚀 Fetching PubMed RCT abstracts (abstract required)")

for topic, query in TOPICS.items():
    print(f"\n📥 {topic.upper()}")

    pmids = fetch_pmids(query, ARTICLES_PER_TOPIC)
    print(f"   🔎 PMIDs found: {len(pmids)}")

    papers = fetch_abstracts(pmids, topic)
    print(f"   ✅ With abstract: {len(papers)}")

    all_papers.extend(papers)
    time.sleep(1)

print(f"\n🎯 TOTAL ARTICLES STORED: {len(all_papers)}")

# =========================
# SAVE RAW DATA
# =========================

Path("200_rct_abstracts.json").write_text(
    json.dumps(all_papers, indent=2, ensure_ascii=False),
    encoding="utf-8"
)


# =========================
# WEAVIATE INGEST (v4 with fallbacks)
# =========================

print("\n🚀 Ingesting into Weaviate (v4 client)")

client = None
try:
    # Try Method 1: Connect with gRPC (recommended)
    print(f"🔌 Attempting connection with gRPC on port {WEAVIATE_GRPC_PORT}...")
    try:
        client = weaviate.connect_to_local(
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=30, query=60, insert=120)
            )
        )
        print(f"✅ Connected to Weaviate with gRPC at localhost:{WEAVIATE_PORT}")
    except Exception as grpc_error:
        print(f"⚠️  gRPC connection failed: {str(grpc_error)[:100]}")
        print(f"🔌 Trying connection with skip_init_checks (REST fallback)...")
        
        # Method 2: Skip gRPC health checks
        client = weaviate.connect_to_local(
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            skip_init_checks=True,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=30, query=60, insert=120)
            )
        )
        print(f"✅ Connected to Weaviate (skip_init_checks) at localhost:{WEAVIATE_PORT}")
    
    # Check if collection exists, delete if it does
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection: {COLLECTION_NAME}")
    
    # Create collection with text2vec-transformers
    collection = client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
        properties=[
            Property(name="pmid", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="abstract", data_type=DataType.TEXT),
            Property(name="topic", data_type=DataType.TEXT),
        ]
    )
    
    print(f"✅ Created collection: {COLLECTION_NAME}")
    
    # Insert papers
    print(f"\n📝 Inserting {len(all_papers)} papers...")
    
    with client.batch.dynamic() as batch:
        for i, paper in enumerate(all_papers, 1):
            batch.add_object(
                collection=COLLECTION_NAME,
                properties=paper
            )
            if i % 50 == 0:
                print(f"   📊 Inserted {i}/{len(all_papers)} papers...")
    
    print(f"✅ Successfully inserted {len(all_papers)} papers into Weaviate")
    
    # Verify count
    papers_collection = client.collections.get(COLLECTION_NAME)
    result = papers_collection.aggregate.over_all(total_count=True)
    print(f"📊 Total papers in collection: {result.total_count}")
    
except Exception as e:
    print(f"❌ Error with Weaviate: {e}")
    print("\n💡 TROUBLESHOOTING:")
    print("   1. Check if Weaviate is running: docker ps")
    print("   2. Restart with proper config: docker-compose down && docker-compose up -d")
    print("   3. Check logs: docker logs medical-rag-weaviate-new")
    print(f"   4. Verify ports {WEAVIATE_PORT} (REST) and {WEAVIATE_GRPC_PORT} (gRPC) are exposed")
    print("\n   See the fixed docker-compose.yml file for proper configuration!")
    
finally:
    if client is not None:
        client.close()
        print("👋 Weaviate connection closed")

print("\n✨ Script completed!")
