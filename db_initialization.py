import sqlite3

# 连接到 SQLite 数据库（如果不存在则创建）
conn = sqlite3.connect('chat_history.db')

# 创建游标对象
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Connections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uri TEXT NOT NULL
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        connection_id INTEGER,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (connection_id) REFERENCES Connections(id)
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        sender TEXT NOT NULL,
        message TEXT NOT NULL,
        sql INTEGER,  
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES Conversations(id)
    )
''')
conn.commit()
conn.close()