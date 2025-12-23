# Weaviate v3 → v4 Migration Guide

## Key Changes in Your Code

### 1. Client Connection
**v3 (OLD):**
```python
client = weaviate.Client(url=f"http://localhost:{WEAVIATE_PORT}")
```

**v4 (NEW):**
```python
client = weaviate.connect_to_local(
    port=WEAVIATE_PORT,
    grpc_port=50051  # default gRPC port
)
```

### 2. Collection Creation
**v3 (OLD):**
```python
client.schema.create_class({
    "class": "ProductionPapers",
    "vectorizer": "text2vec-transformers",
    # ...
})
```

**v4 (NEW):**
```python
collection = client.collections.create(
    name="ProductionPapers",
    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
    properties=[
        Property(name="pmid", data_type=DataType.TEXT),
        Property(name="title", data_type=DataType.TEXT),
        Property(name="abstract", data_type=DataType.TEXT),
        Property(name="topic", data_type=DataType.TEXT),
    ]
)
```

### 3. Batch Insertion
**v3 (OLD):**
```python
client.batch.configure(batch_size=100)
with client.batch as batch:
    for paper in all_papers:
        batch.add_data_object(paper, "ProductionPapers")
```

**v4 (NEW):**
```python
with client.batch.dynamic() as batch:
    for paper in all_papers:
        batch.add_object(
            collection="ProductionPapers",
            properties=paper
        )
```

### 4. Querying (for later use)
**v3 (OLD):**
```python
result = client.query.get("ProductionPapers", ["title", "abstract"]).do()
```

**v4 (NEW):**
```python
collection = client.collections.get("ProductionPapers")
result = collection.query.fetch_objects(limit=10)
```

### 5. Connection Cleanup
**v4 IMPORTANT:**
```python
# Always close the connection when done
client.close()
```

## Alternative: Pin to v3 if needed

If you need to continue using v3 code (not recommended):

```bash
pip uninstall weaviate-client
pip install "weaviate-client>=3.26.7,<4.0.0"
```

Then change v3 syntax:
```python
client = weaviate.Client(url=f"http://localhost:{WEAVIATE_PORT}")
```

## Benefits of v4
- Better performance with gRPC
- Type safety
- Cleaner API
- Better error handling
- Automatic connection management

## Additional Resources
- [Python Client v4 docs](https://weaviate.io/developers/weaviate/client-libraries/python)
- [v3→v4 Migration guide](https://weaviate.io/developers/weaviate/client-libraries/python/v3_v4_migration)
