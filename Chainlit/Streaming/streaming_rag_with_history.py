## Todo add conversation memory

from operator import itemgetter

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory

from operator import itemgetter

import pathlib
import json
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():

    mem = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("memory", mem)

    current_dir = pathlib.Path(__file__).parent.resolve()
    loader = DirectoryLoader(str(current_dir) + '/sources', glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    for doc in docs:
        doc_name = (doc.metadata['source'].split('/')[-1].split('.')[0])
        doc.metadata['Law Name'] = doc_name 
        doc.metadata['Alt Law Name'] = doc_name.split('_')[1]
        del doc.metadata['source']
        print(doc.metadata)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    for split in splits:
        header = ""
        for key, val in split.metadata.items():
            header += f"{key}: {val}, "
        header = "An excpert from, " + header[:-2] + "\n-----\n"
        split.page_content = header + split.page_content
    
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def seralized_retriever(*args, **kwargs):
        question = args[0]['question']
        history = mem.load_memory_variables({})['history']
        # print("RETRIEVER HISTORY")
        # print(history[-2:])
        first_message_and_response = [item.content for item in history[-2:]]
        # print(first_message_and_response)
        docs = retriever.get_relevant_documents(f"{question}, History: {first_message_and_response}")
        retrieved_docs = [doc.page_content for doc in docs]
        cl.user_session.set("retrieved_docs", retrieved_docs)
        return retrieved_docs

    template = """Answer the question based on only the following context and message history.

    Context: {context}
    
    Message History: {history}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", temperature=0.7, max_tokens=1000, streaming=True)
    
    chain = (
        {"context": seralized_retriever, "question": RunnablePassthrough(), "history": RunnableLambda(mem.load_memory_variables) | itemgetter("history")}
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):

    mem = cl.user_session.get("memory")
    print(mem.load_memory_variables({}))
    

    chain = cl.user_session.get("chain")  # type: Runnable
    response = cl.Message(content="")

    async for chunk in chain.astream({"question": message.content}, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        await response.stream_token(chunk)
        # print(chunk)

    await response.send()
    # print(message.content, response.content)
    mem.save_context({"input": message.content}, {"output": response.content})

    # print(cl.user_session.get("retrieved_docs"))
    text_elements = []  # type: List[cl.Text]
    source_documents = cl.user_session.get("retrieved_docs")
    for source_idx, source_doc in enumerate(source_documents):
        source_name = f"Source {source_idx + 1}"
        # Create the text element referenced in the message
        text_elements.append(
            cl.Text(content=source_doc, name=source_name)
        )
    source_names = [text_el.name for text_el in text_elements]
    answer = ""

    if source_names:
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()