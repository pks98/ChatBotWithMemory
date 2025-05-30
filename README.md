# Legal RAG ChatBot

An advanced Legal RAG (Retrieval-Augmented Generation) ChatBot that uses LangChain and OpenAI to provide intelligent legal document analysis and question answering.

## Features

- Document loading and processing (PDF support)
- Advanced question decomposition
- Smart context retrieval with duplicate removal
- Step-by-step reasoning with source citations
- Memory-efficient document chunking
- FAISS vector store for efficient similarity search

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pks98/ChatBotWithMemory.git
cd ChatBotWithMemory
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```python
from legal_rag_bot import LegalRAGBot

# Initialize the bot
bot = LegalRAGBot()

# Load your legal document
bot.load_document("path_to_your_document.pdf")

# Process the documents
bot.process_documents()

# Ask questions
question = "What is the effective date of the agreement? Who are the parties to the agreement?"
answer = bot.query(question)
print(f"Answer: {answer}")
```

## Project Structure

- `legal_rag_bot.py`: Main implementation of the Legal RAG Bot
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)
- `.gitignore`: Git ignore rules
