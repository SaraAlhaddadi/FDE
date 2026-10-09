import os
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "enterprise_knowledge_v2"

def init_advanced_rag():
    """تهيئة قاعدة البيانات المتجهية بالسياسات والتعليمات المتقدمة"""
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    documents = [
        {"id": 1, "text": "السياسة المالية 2026: جميع التسويات البنكية التي تتجاوز قيمتها 10,000 ريال سعودي تتطلب موافقة مزدوجة من المدير المالي."},
        {"id": 2, "text": "سياسة الموارد البشرية 2026: تقييم أداء الموظفين يتم بشكل نصف سنوي في شهري يونيو وديسمبر. يحصل الموظفون المتميزون على مكافأة تصل إلى 15%."},
        {"id": 3, "text": "سياسة أمن المعلومات: يجب تغيير كلمات المرور كل 90 يوماً. وتُخزن مفاتيح الـ API حصراً في مدير المفاتيح المعتمد (Secrets Manager)."}
    ]

    embed_response = co.embed(
        texts=[doc["text"] for doc in documents],
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    embeddings = embed_response.embeddings.float_

    points = [
        PointStruct(id=doc["id"], vector=embeddings[i], payload={"text": doc["text"]})
        for i, doc in enumerate(documents)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ [RAG Engine]: Vector DB Initialized with Multilingual Embeddings.")

def query_knowledge_base(query: str) -> dict:
    """استرجاع المستندات مع إعادة الترتيب المتقدم بـ Cohere Rerank"""
    query_embed = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings.float_[0]

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embed,
        limit=3
    )
    raw_docs = [hit.payload["text"] for hit in results]

    if not raw_docs:
        return {"retrieved_policies": []}

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=raw_docs,
        top_n=2
    )

    reranked_docs = [raw_docs[res.index] for res in rerank_response.results]
    return {"retrieved_policies": reranked_docs}