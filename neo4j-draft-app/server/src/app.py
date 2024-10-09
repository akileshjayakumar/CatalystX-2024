import os
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    logger.info("Initialized OpenAI LLM: %s", llm)
except Exception as e:
    logger.error("Failed to initialize OpenAI LLM: %s", e)
    raise

try:
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME,
                       password=NEO4J_PASSWORD)
    logger.info("Initialized Neo4j graph connection: %s", graph)
except Exception as e:
    logger.error("Failed to initialize Neo4j graph connection: %s", e)
    raise

SESSION_ID = str(uuid4())
logger.info("Generated session ID: %s", SESSION_ID)


def get_memory(session_id):
    logger.info("Getting memory for session ID: %s", session_id)
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI tutor for Singaporean students, knowledgeable about the Singapore curriculum and 21st-century skills.\n"
                   "Your goal is to provide curriculum-aligned learning experiences while promoting critical thinking, creativity, and cyber wellness.\n"
                   "Current subject: {subject}\n"
                   "Current topic: {topic}\n\n"
                   "Please provide responses that:\n"
                   "1. Are accurate and aligned with the Singapore curriculum\n"
                   "2. Encourage critical thinking and problem-solving\n"
                   "3. Foster creativity and innovation\n"
                   "4. Promote digital literacy and cyber wellness\n"
                   "5. Are appropriate for the student's education level: {educationLevel}\n\n"
                   "Remember to be encouraging and supportive in your responses."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

chat_chain = chat_prompt | llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

logger.info("Initialized chat with message history: %s",
            chat_with_message_history)


class ChatRequest(BaseModel):
    question: str
    subject: str
    topic: str
    educationLevel: str
    context: str


@app.post("/chat/")
def chat(request: ChatRequest):
    logger.info("Received chat request: %s", request)

    try:
        chat_history = get_memory(SESSION_ID)

        input_data = {
            "question": request.question,
            "context": request.context,
            "subject": request.subject,
            "topic": request.topic,
            "educationLevel": request.educationLevel,
            "chat_history": chat_history
        }

        config = {
            "configurable": {
                "session_id": SESSION_ID
            }
        }

        response = chat_with_message_history.invoke(
            input=input_data, config=config)

        logger.info("Generated response: %s", response)
        return {"response": response}

    except Exception as e:
        logger.error("Error processing chat request: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}")
