# 🚀 Quick Reference - Команды для запуска

## Последовательность действий

### 1️⃣ Подготовка (один раз)
```powershell
cd C:\Users\nnazg\Documents\advanced_rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Docker (каждый раз)
```powershell
# Остановить старые
docker stop medical-rag-weaviate-new
docker rm medical-rag-weaviate-new

# Запустить новые
docker-compose up -d

# Проверить статус (должно быть "healthy")
docker ps

# Дождаться готовности
docker logs -f medical-rag-weaviate-fixed
# Ждем: "Weaviate is ready to rock!"
```

### 3️⃣ Загрузка данных (если нужно)
```powershell
# Опция A: Использовать существующий
# (уже есть 200_rct_abstracts.json - ничего не делать)

# Опция B: Загрузить заново (облегченная версия, 50 вместо 200)
python fetch_200_abstracts_robust.py
```

### 4️⃣ Baseline (первый запуск)
```powershell
python evaluate_baseline.py

# Результат: baseline_results.json
# Время: ~3-5 минут для 5 вопросов
```

### 5️⃣ Enhanced (второй запуск)
```powershell
python evaluate_enhanced.py

# Результат: enhanced_results.json
# Автоматически сравнит с baseline
# Время: ~5-10 минут для 5 вопросов
```

### 6️⃣ Визуализация (финал)
```powershell
python visualize_results.py

# Результат:
# - comparison.png
# - detailed_report.txt
```

---

## 🔥 Быстрый тест (3 минуты)

Если хотите быстро проверить что все работает:

```powershell
# 1. Docker up
docker-compose up -d && docker ps

# 2. Quick test baseline (3 вопроса)
# Отредактируйте evaluate_baseline.py:
# TEST_QUESTIONS = TEST_QUESTIONS[:3]
python evaluate_baseline.py

# 3. Quick test enhanced
# Отредактируйте evaluate_enhanced.py:
# (то же самое)
python evaluate_enhanced.py
```

---

## 📋 Проверка перед сдачей

```powershell
# Все ли файлы на месте?
ls README.md
ls evaluate_baseline.py
ls evaluate_enhanced.py
ls baseline_results.json
ls enhanced_results.json
ls comparison.png

# Все ли контейнеры healthy?
docker ps | grep healthy
# Должно показать weaviate, transformers, ollama

# Работает ли Ollama?
curl http://localhost:11434/api/version

# Работает ли Weaviate?
curl http://localhost:8082/v1/schema
```

---

## 🆘 Быстрые решения проблем

### Docker не стартует
```powershell
docker-compose down -v
docker-compose up -d --force-recreate
```

### gRPC ошибка
```powershell
# Проверьте порт 50051
netstat -ano | findstr "50051"

# Если занят, измените в docker-compose.yml:
#   - "50052:50051"
# И в скриптах: grpc_port=50052
```

### Out of memory
```python
# В evaluate_*.py измените:
TEST_QUESTIONS = TEST_QUESTIONS[:2]  # Только 2 вопроса
```

### RAGAS слишком медленно
```python
# Используйте меньший LLM для evaluation
# В evaluate_*.py добавьте:
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2:1b")  # Меньшая модель
```

---

## 📊 Что должно получиться

### baseline_results.json
```json
{
  "timestamp": "2025-12-23 ...",
  "metrics": {
    "faithfulness": 0.75,
    "answer_relevancy": 0.65,
    "context_precision": 0.55,
    "context_recall": 0.70,
    "answer_correctness": 0.60
  },
  "summary": {
    "avg_score": 0.65,
    "test_size": 5
  }
}
```

### enhanced_results.json (лучше)
```json
{
  "timestamp": "2025-12-23 ...",
  "metrics": {
    "faithfulness": 0.82,
    "answer_relevancy": 0.85,    // +30%!
    "context_precision": 0.77,   // +40%!
    "context_recall": 0.75,
    "answer_correctness": 0.70
  },
  "summary": {
    "avg_score": 0.78,
    "test_size": 5
  }
}
```

### Консольный вывод
```
BASELINE vs ENHANCED COMPARISON
================================

ANSWER_RELEVANCY:
  Baseline:  0.650
  Enhanced:  0.845
  Change:    +30.0%
  ✅ TARGET ACHIEVED (≥30%)

CONTEXT_PRECISION:
  Baseline:  0.550
  Enhanced:  0.770
  Change:    +40.0%
  ✅ TARGET ACHIEVED (≥30%)

🎯 SUCCESS: 2 metrics improved by ≥30%!
```

---

## 🎯 Что главное для 100/100

1. ✅ **README.md** - kick-off report с rationale
2. ✅ **Automated tests** - evaluate_*.py scripts
3. ✅ **≥30% improvement** - минимум 1 метрика
4. ✅ **Advanced techniques** - reranking, hybrid search
5. ✅ **Clear documentation** - все объяснено

---

## ⏱️ Timeline

| Шаг | Время | Команда |
|-----|-------|---------|
| Setup | 5 мин | `pip install -r requirements.txt` |
| Docker | 2 мин | `docker-compose up -d` |
| Baseline | 5 мин | `python evaluate_baseline.py` |
| Enhanced | 10 мин | `python evaluate_enhanced.py` |
| Visualize | 1 мин | `python visualize_results.py` |
| **TOTAL** | **23 мин** | |

---

**Готово! Все команды под рукой 🎯**
