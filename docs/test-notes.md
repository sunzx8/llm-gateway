```
curl -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
http://127.0.0.1:8000/llm/v1/models 

```


```bash 
curl -X POST http://127.0.0.1:8000/llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "我是一名初中二年级的学生，你是谁，可以帮我写作业吗？"}
    ]
  }' 

curl -X POST http://127.0.0.1:8000/llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
  -d '{
    "model": "fake-model",
    "messages": [
      {"role": "user", "content": "我是一名初中二年级的学生，你是谁，可以帮我写作业吗？"}
    ]
  }' 

curl -X POST http://127.0.0.1:8000/llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
  -d '{
    "model": "fake-model",
    "messages": [
      {"role": "user", "content": "你是谁？"}
    ],
    "stream": true
  }' 
```

```bash
curl -X POST http://127.0.0.1:8000/llm/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "你是谁？"}
    ],
    "stream": true
  }'
  ```


python3 -m eval.eval_script --benchmarks personamem --data-dir dataset/personamem --split 32k --conv-concurrency 1