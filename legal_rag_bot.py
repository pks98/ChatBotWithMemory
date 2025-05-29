import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class LegalRAGBot:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """Initialize the Legal RAG Bot with specified model."""
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.documents = []
        
    def load_document(self, pdf_path: str) -> None:
        """Load and process a PDF document."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Add metadata for source tracking
        for page in pages:
            page.metadata.update({
                "source": os.path.basename(pdf_path),
                "page": page.metadata.get("page", 0) + 1
            })
        
        self.documents.extend(pages)
        
    def process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Process documents into chunks and create vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(self.documents)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
    def break_down_question(self, question: str) -> List[str]:
        """Break down the main question into sub-questions for better retrieval."""
        breakdown_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given the following question, break it down into 2-3 specific sub-questions that would help answer it comprehensively.
            Original Question: {question}
            
            Provide the sub-questions in a clear, numbered format.
            Focus on different aspects of the question that need to be addressed."""
        )
        
        # Use the new LCEL pattern instead of LLMChain
        chain = breakdown_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        # Parse the result to get individual questions
        sub_questions = [q.strip() for q in result.split('\n') if q.strip() and q[0].isdigit()]
        return sub_questions

    def get_unique_context(self, questions: List[str]) -> str:
        """Get unique context chunks for multiple questions."""
        all_chunks = []
        seen_chunks = set()
        
        for question in questions:
            docs = self.vector_store.similarity_search(question, k=3)
            for doc in docs:
                # Create a unique identifier for the chunk
                chunk_id = f"{doc.page_content[:100]}"
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_chunks.append(doc)
        
        # Format the unique chunks with their sources
        formatted_chunks = []
        for doc in all_chunks:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted_chunks.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
        
        return "\n\n".join(formatted_chunks)

    def create_chain(self) -> Any:
        """Create the advanced RAG chain with step-by-step reasoning."""
        template = """You are a legal expert assistant. Use the following pieces of context to answer the question at the end.
        
        Context: {context}
        
        Question: {question}
        
        Your step-by-step reasoning and answer (with sources):
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain using LCEL pattern
        chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, question: str) -> str:
        """Query the RAG system with advanced reasoning."""
        if not self.vector_store:
            raise ValueError("No documents have been processed. Please load and process documents first.")
        
        # Break down the question into sub-questions
        sub_questions = self.break_down_question(question)
        
        # Get unique context for all sub-questions
        context = self.get_unique_context(sub_questions)
        
        # Create and run the chain
        chain = self.create_chain()
        return chain.invoke({"context": context, "question": question})

def main():
    # Example usage
    bot = LegalRAGBot()
    
    # Load your legal document
    pdf_path = "D:\Agents\AdvanceRAGChatBot\SERVICE AGREEMENT-2.pdf"  # Replace with actual path
    bot.load_document(pdf_path)
    
    # Process the documents
    bot.process_documents()
    
    # Example query
    question = "What is the effective date of the agreement? Who are the parties to the agreement?"
    answer = bot.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main() 