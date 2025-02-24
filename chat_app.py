import uuid

import os
import tempfile
import streamlit as st
from streamlit_chat import message
import chromadb

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# Chromaのキャッシュをリセット
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Streamlitによって、タイトル部分のUIをの作成
st.title("Chatbot in Streamlit")

# LLMのインスタンスの作成
llm = ChatOpenAI(
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        model_name=os.environ["OPENAI_API_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        max_retries=5,
        timeout=60,
    )

# プロジェクト内のPDFファイルを読み込み
file_path = os.environ["DOCUMENT_FILE_PATH"]
loader = PyPDFLoader(file_path)
docs = loader.load()

# 設定を定義
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 64
PERSIST_PATH = "./.chroma_db"
COLLECTION_NAME = "langchain"

# テキスト分割設定
text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = CHUNK_SIZE,
        chunk_overlap  = CHUNK_OVERLAP,
    )

# ドキュメントデータを分割
docs = text_splitter.split_documents(docs)

# 分割したデータをベクトル化
vectorstore = Chroma.from_documents(
    collection_name=COLLECTION_NAME,
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory=PERSIST_PATH
)

# ベクトルストアからレトリーバーを作成
retriever = vectorstore.as_retriever()
st.session_state["retriever"] = retriever
retriever = st.session_state["retriever"]

if "llm" not in st.session_state:
    st.session_state["llm"] = llm
llm = st.session_state["llm"]

# セッション内に保存されたチャット履歴のメモリの取得
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
session_id = st.session_state["session_id"]

if "history" not in st.session_state:
    st.session_state["history"] = ChatMessageHistory()
history = st.session_state["history"]

def get_session_history(session_id: str):
    '''
    #sessionから履歴を取得
    '''
    return history

def create_chain():
    '''
    RAGを実行し、過去の文脈を考慮した回答を生成
    '''

    # 履歴のシステムプロンプト
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    # Contextualizing用のプロンプトテンプレートの作成
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 履歴レトリーバーの作成
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # 回答のシステムプロンプト
    qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer \
        the question. If you don't know the answer, say that you \
        don't know. Use three sentences maximum and keep the \
        answer concise.
        {context}"""

    # 実際にRAGを実行するためのQA用プロンプトテンプレートの作成
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 質問回答チェーン
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 履歴レトリーバーと質問回答チェーンを順番に適用するチェーンを作成
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    return conversational_rag_chain

if "chain" not in st.session_state:
    # create_chain関数を実行し、回答を生成
    st.session_state["chain"] = create_chain()

chain = st.session_state["chain"]

# 履歴を表示
for h in history:
    for message in h[1]:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

# ユーザーの入力に対しAIが回答する
if prompt := st.chat_input("質問を入力"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = chain.stream(
            {"input": prompt},
            config={
                "configurable": {"session_id": session_id},
            },
        )

        def s():
            for chunk in stream:
                if "answer" in chunk:
                    yield chunk["answer"]

        st.write_stream(s)

