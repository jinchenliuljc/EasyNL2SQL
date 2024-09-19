from agent import main
import fastapi
import sqlite3
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from summary_agent import summary_agent
import chromadb
import json
import os
import uuid
import pandas as pd

message_history_uri = 'chat_history.db'
knowledge_path = "./knowledgebase"

app = FastAPI()


class Connection(BaseModel):
    uri: str


class Conversation(BaseModel):
    connection_id: int


class Message(BaseModel):
    conversation_id: int
    sender: str
    message: str


class MessageOut(BaseModel):
    sender: str
    message: str


def get_history_connection():
    conn = sqlite3.connect(message_history_uri)
    # conn.row_factory = sqlite3.Row  # 可选: 使查询结果更便于字典访问
    return conn


@app.get("/api/conversation_history")
def get_message_history(conversation_id):
    with get_history_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
        SELECT sender, message FROM Messages WHERE conversation_id = ?
        ORDER BY timestamp ASC
        ''', (conversation_id,))

        messages = cursor.fetchall()
        print(f'history_retrieved: \n {messages}')

    return messages


def query_history(sql):
    with get_history_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(sql)

        results = cursor.fetchall()
    
    return results


def insert_history(sql):
    with get_history_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        id = cursor.lastrowid
        conn.commit()
    return id


def get_embedding(text, model="text-embedding-3-small"):
    from openai import OpenAI
    import dotenv

    dotenv.load_dotenv()
    openai_client = OpenAI()
    text = text.replace("\n", " ")

    return openai_client.embeddings.create(input = [text], model=model).data[0].embedding


def merge_knowledge(value):
    existing_value = query_knowledge(value)
    value = merge_agent(existing_value)

    return value


@app.get("/api/insert_knowledge")
def insert_knowledge(key, value, metadata):
    if not os.path.exists('./test'):
        os.mkdir('test')

    client = chromadb.PersistentClient(path=knowledge_path)

    collection = client.get_or_create_collection(name="knowledge", metadata={"hnsw:space": "ip"})

    # collection.peek()
    # value = merge(value)

    vec = get_embedding(key)


    id = str(uuid.uuid1().int)

    if isinstance(metadata,str):
        metadata = json.loads(metadata)

    collection.add(
        documents=[json.dumps({'user_query_summary':key, 'ai': value})],
        embeddings=[vec],
        metadatas=[metadata],
        ids=[id]
    )


@app.get("/api/query_knowledge")
def query_knowledge(query):
    client = chromadb.PersistentClient(path=knowledge_path)

    collection = client.get_or_create_collection(name="knowledge", metadata={"hnsw:space": "ip"})

    res = collection.query(get_embedding(query), n_results=3)

    qas = pd.DataFrame({
    'ids': res['ids'][0],
    'documents': res['documents'][0],
    'metadatas': res['metadatas'][0],
    'distances': res['distances'][0]
    }).sort_values(by='distances')['documents'].values.tolist()

    return qas


# @app.get("/api/conversation_history")#response_model=List[MessageOut])
# def message_history(conversation_id: int):
#     messages = get_message_history(conversation_id)
#     # return [{message[0]:message[1]} for message in messages]
#     return messages



@app.post("/api/start_conversation")
def new_chat(connection_id):
    with get_history_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO Conversations (connection_id, start_time) VALUES (?, CURRENT_TIMESTAMP)', (connection_id,))
        conversation_id = cursor.lastrowid
        conn.commit()
    

    return {'id': conversation_id}


@app.post("/api/send_message")
def chat(input, conversation_id):
    with get_history_connection() as conn:
        cursor = conn.cursor()
        history = get_message_history(conversation_id)
        cursor.execute('''
        INSERT INTO Messages (conversation_id, sender, message)
        VALUES (?, ?, ?)
        ''', (conversation_id, 'human', input)) #input 类型限制必须是文本

        conn.commit()
    
    res = main(input, history)
    

    try:
        output = {"sql":res['SQL'], "knowledge":res['Knowledge'], "thought":res['Thought']}
        # sql, knowledge, thought = (res['SQL'], res['Knowledge'], res['Thought'])
        json_output = json.dumps(output)
        with get_history_connection() as conn:
            cursor = conn.cursor()
            history = get_message_history(conversation_id)
            cursor.execute('''INSERT INTO Messages (conversation_id, sender, message, sql)
            VALUES (?, ?, ?, ?)
            ''', (conversation_id, 'ai', json_output, 1)) #input 类型限制必须是文本
            conn.commit()

    # sql, knowledge, thoughts = (output['SQL'], output['Knowledges'], output['Thoughts'])
        return output
    
    except Exception:
        output = res['output']
        with get_history_connection() as conn:
            cursor = conn.cursor()
            history = get_message_history(conversation_id)
            cursor.execute('''
            INSERT INTO Messages (conversation_id, sender, message,sql)
            VALUES (?, ?, ?, ?)
            ''', (conversation_id, 'ai', output, 0)) #input 类型限制必须是文本

            conn.commit()
        
        return output
    # if not res==None:
    #     cursor.execute('''
    #     INSERT INTO Messages (conversation_id, sender, message)
    #     VALUES (?, ?, ?)
    #     ''', (conversation_id, 'ai', res['content'])) #content or other things
        
    # submit_sql直接接收session_id存入message_history库

@app.post("/api/verify")
def verify(conversation_id):
    history = get_message_history(conversation_id)
    summary = summary_agent(history)
    key = json.loads(summary)['summary']

    value_query = f'''SELECT message 
        FROM Messages 
        WHERE ("sql" = {1}) AND ("conversation_id" = {conversation_id})
        ORDER BY timestamp DESC 
        LIMIT 1
            '''
    
    value = query_history(value_query)[0][0]

    db_id_query = f'''SELECT connection_id 
        FROM Conversations c 
        WHERE id = {conversation_id}
            '''
    
    db_id = query_history(db_id_query)[0][0]

    metadata = {'id':db_id}


    insert_knowledge(key, value, metadata)


@app.post("/api/create_connection")
def add_db(uri):
    with get_history_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO Connections (uri) VALUES (?)', (uri,))
        conn.commit()
        connection_id = cursor.lastrowid
    return {'id': connection_id}




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)

