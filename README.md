```markdown
# NGAI DeepThink Chatbot Backend

A high-performance FastAPI backend that simulates AI "thinking" with step-by-step reasoning and real-time streaming. Features hybrid HTTP polling + WebSocket architecture for optimal user experience.

## 🌟 Features

- **Step-by-Step Thinking Simulation**: Shows AI reasoning process like DeepSeek/Chain-of-Thought
- **Real-time Streaming**: Character-by-character answer streaming with WebSocket
- **Hybrid API**: Both HTTP polling (for mobile/simple clients) and WebSocket (for web frontend)
- **Single AI Call Optimization**: One API call for both thinking steps and final answer
- **Memory Management**: Automatic cleanup with TTL caches
- **Production Ready**: Rate limiting, API key validation, CORS, logging, error handling

## 📁 Project Structure

```

ngai-backend/
├──main.py              # FastAPI application
├──requirements.txt     # Python dependencies
├──.env                # Environment variables
├──README.md           # This file
└──(optional: Dockerfile, .gitignore)

```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- [OpenRouter API Key](https://openrouter.ai/keys)

### 2. Installation

```bash
# Clone repository (if applicable)
# git clone <your-repo>
# cd ngai-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Configuration

Create .env file:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
API_KEYS=client1-secret-key,client2-secret-key

# Optional
PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://yourapp.com
CHUNK_SIZE=4
HEARTBEAT_INTERVAL=10
```

4. Run the Server

```bash
# Development (auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

Server will start at: http://localhost:8000

📚 API Documentation

Once running, visit:

· Swagger UI: http://localhost:8000/docs
· ReDoc: http://localhost:8000/redoc

🔌 API Endpoints

HTTP Polling API

1. Start Thinking Process

```http
POST /chat/start
Content-Type: application/json
x-api-key: your-client-key
```

Request:

```json
{
  "user_id": "user123",
  "message": "What is quantum computing?",
  "model": "openassistant-medium"
}
```

Response:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

2. Check Thinking Status

```http
GET /chat/status/{task_id}
x-api-key: your-client-key
```

Response:

```json
{
  "status": "analyzing",
  "step": 2,
  "message": "Analyzing the context and keywords...",
  "progress": 28,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

3. Get Final Result

```http
GET /chat/result/{task_id}
x-api-key: your-client-key
```

Response:

```json
{
  "thinking_process": [
    {"status": "analyzing", "step": 1, "message": "...", "progress": 14, "timestamp": "..."},
    {"status": "reasoning", "step": 2, "message": "...", "progress": 28, "timestamp": "..."}
  ],
  "final_answer": "Quantum computing uses quantum bits...",
  "confidence": 0.95,
  "citations": [],
  "suggested_followups": ["How does quantum superposition work?", "What are qubits?"]
}
```

4. Regenerate Response

```http
POST /chat/regenerate
Content-Type: application/json
x-api-key: your-client-key
```

Request:

```json
{
  "user_id": "user123",
  "last_task_id": "previous-task-id"
}
```

Response:

```json
{
  "task_id": "new-task-id"
}
```

WebSocket API

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat', {
  headers: { 'x-api-key': 'your-client-key' }
});

// Send initial message
ws.send(JSON.stringify({
  message: "Explain quantum computing",
  model: "openassistant-medium"
}));
```

WebSocket Message Types:

Type Description Example
connected Connection established {"type": "connected", "task_id": "..."}
thinking_step Thinking step update {"type": "thinking_step", "step": {...}}
answer_chunk Answer character chunk {"type": "answer_chunk", "chunk": "Qua"}
complete Final answer ready {"type": "complete", "final_answer": "..."}
ping Heartbeat {"type": "ping"}
error Error message {"type": "error", "detail": "..."}

⚙️ Configuration Details

Environment Variables

Variable Required Default Description
OPENROUTER_API_KEY Yes - Your OpenRouter API key
API_KEYS Yes - Comma-separated client API keys
PORT No 8000 Server port
LOG_LEVEL No INFO Logging level
CORS_ORIGINS No  *  Comma-separated allowed origins
CHUNK_SIZE No 4 Characters per streaming chunk
HEARTBEAT_INTERVAL No 10 WebSocket heartbeat interval (seconds)

Available AI Models

The backend supports these free OpenRouter models:

· openassistant-medium
· openassistant-large
· openassistant-mini

📊 Performance Optimization

Single AI Call Architecture

```
Traditional: User → AI Call 1 (Steps) → AI Call 2 (Answer) → Response
This Backend: User → Single AI Call → Parse Steps & Answer → Stream Locally
```

Memory Management

· Thinking cache: 2000 tasks max, 1 hour TTL
· User history: 10000 users max, 1 hour TTL
· Automatic cleanup of expired entries

🔒 Security

· API Key Validation: All requests require valid API key
· Rate Limiting: 10 requests/minute per IP
· Input Validation: Pydantic models with length/size limits
· Content Filtering: Basic toxic content detection
· CORS: Configurable origins (restrict in production)

🐳 Docker Deployment

Create Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ngai-backend .
docker run -p 8000:8000 --env-file .env ngai-backend
```

🧪 Testing

Health Check

```bash
curl http://localhost:8000/health
```

List Models

```bash
curl http://localhost:8000/models
```

Example Chat (using HTTP)

```bash
curl -X POST http://localhost:8000/chat/start \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-client-key" \
  -d '{"user_id": "test", "message": "Hello", "model": "openassistant-medium"}'
```

🚨 Error Handling

Status Code Description
400 Invalid request (bad model, toxic content, etc.)
403 Invalid API key
404 Task not found
429 Rate limit exceeded
500 Internal server error
502 AI service error
504 AI service timeout

📈 Monitoring

Headers

· X-Request-ID: Unique request identifier
· X-Process-Time: Request processing time in seconds

Logging

· Structured logging with timestamps and levels
· Request/response logging
· Error stack traces
· WebSocket connection events

🔄 Regeneration Feature

The /chat/regenerate endpoint:

1. Looks up the original message from user history
2. Creates a new thinking task with same parameters
3. Returns new task_id for tracking

🎯 Frontend Integration Examples

React Component (WebSocket)

```javascript
import { useEffect, useState } from 'react';

function ChatInterface() {
  const [thinkingSteps, setThinkingSteps] = useState([]);
  const [answer, setAnswer] = useState('');
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/chat', {
      headers: { 'x-api-key': 'your-key' }
    });
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch(data.type) {
        case 'thinking_step':
          setThinkingSteps(prev => [...prev, data.step]);
          break;
        case 'answer_chunk':
          setAnswer(prev => prev + data.chunk);
          break;
        case 'complete':
          console.log('Complete:', data.final_answer);
          break;
      }
    };
    
    return () => ws.close();
  }, []);
  
  // ... render UI
}
```

Mobile App (Polling)

```javascript
// React Native / Flutter example pattern
async function chatWithPolling(message) {
  // 1. Start task
  const start = await fetch('/chat/start', {...});
  const { task_id } = await start.json();
  
  // 2. Poll for updates
  const poller = setInterval(async () => {
    const status = await fetch(`/chat/status/${task_id}`);
    const data = await status.json();
    
    // Update UI with thinking step
    updateThinkingUI(data);
    
    if (data.status === 'complete') {
      clearInterval(poller);
      const result = await fetch(`/chat/result/${task_id}`);
      // Display final answer
    }
  }, 500);
}
```

🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

📄 License

This project is licensed under the MIT License.

📞 Support

For questions or issues:

· Email: niishaldas@gmail.com
· GitHub Issues: [Repository Issues]
· Documentation: /docs endpoint on running server

🌐 Live Demo

Visit the API documentation at http://your-server:8000/docs when deployed.

---

Built with ❤️ using FastAPI, OpenRouter, and modern async Python

```

## Your 3 Files:

### 1. `requirements.txt`
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.1
pydantic==2.5.0
cachetools==5.3.2
slowapi==0.1.8
python-dotenv==1.0.0
```

2. .env

```bash
# OpenRouter API Key (get from https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Your client API keys (comma-separated)
API_KEYS=client1-secret-key,client2-secret-key

# Optional settings
PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://yourapp.com
CHUNK_SIZE=4
HEARTBEAT_INTERVAL=10
```

3. main.py (your final optimized version)

Now you have a complete project with:

· ✅ README.md with full documentation
· ✅ requirements.txt with dependencies
· ✅ .env template with configuration
· ✅ main.py with optimized code
