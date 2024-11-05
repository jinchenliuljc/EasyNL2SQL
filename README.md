# 🚀 EasyNL2SQL -- A Database Agent
A toy Database agent with ability to explore database with toolcalling, able to reflect and has long/short term memory. 
### install requirements
``` git clone --depth 1 https://github.com/jinchenliuljc/AVIDA.git
cd AVIDA
pip install -r requirements.txt
```

### create .env file
Use your text editor, e.g. VScode, to create a .env file. Enter envronment variables like this
```
OPENAI_API_KEY= your openai api (only support this sofar)
LANGCHAIN_API_KEY = your langchain api, you can use langsmith to monitor the agent flow. For details, check their docs.
LANGCHAIN_TRACING_V2 = true
LANGCHAIN_PROJECT = xxxxxx
```

### create chat_history database
`python db_initialization.py`

### link your database
Because i am lazy i did not write an interface for you to modify your database uri. Go agent.py and search "填入数据库连接信息", and type your database uri there. Only support sqlite so far.

To run the [example](https://github.com/jinchenliuljc/AVIDA/blob/main/agent_client.ipynb), use [Chinook.db](https://www.sqlitetutorial.net/sqlite-sample-database/)


### run backend.py
`python backend.py`

### try the agent with agent_client.py
see [agent_client.py](https://github.com/jinchenliuljc/AVIDA/blob/main/agent_client.ipynb)
