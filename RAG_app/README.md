# Local RAG App

Package này triển khai pipeline local RAG cho dữ liệu Bệnh viện Hạnh Phúc.

## Cài dependency

```powershell
python -m pip install -r requirements-rag.txt
```

## Biến môi trường

Tạo `.env` ở root project hoặc export trực tiếp:

```env
GEMINI_API_KEY=your_api_key
GEMINI_GENERATION_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=hanhphuc_hospital_rag
```

Nếu không có Docker/Qdrant server, có thể dùng embedded local Qdrant:

```env
QDRANT_PATH=Data_RAG/index/qdrant_local
QDRANT_COLLECTION=hanhphuc_hospital_rag
```

## Lệnh chính

```powershell
python -m RAG_app.cli ingest --source hanhphuc --recreate
python -m RAG_app.cli ask "gói IVF Standard gồm gì"
python -m RAG_app.cli ask "bác sĩ nào hỗ trợ sinh sản" --json
python -m RAG_app.cli eval --qa Data_RAG/qa/hanhphuc_rag_qa_200.jsonl --limit 10
```

V1 chỉ ingest `Data_RAG/entities/hanhphuc_entities.jsonl`. File Q&A dùng để eval, không index vào vector DB.
