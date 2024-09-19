# AVIDA-A Very Intelligent Data Agent
## Get started
### install requirements
``` git clone --depth 1 https://github.com/jinchenliuljc/AVIDA.git
cd AVIDA
pip install -r requirements.txt
```

### create .env file
Use your text editor, e.g. VScode, to create a .env file （. requirements.txt). Enter envronment variables like this
```
OPENAI_API_KEY= your openai api (only support this sofar)
LANGCHAIN_API_KEY = your langchain api
LANGCHAIN_TRACING_V2 = true
LANGCHAIN_PROJECT = xxxxxx
```

### run backend.py
`python backend.py`

### try the agent with agent_client.py
(. agent_client.py)
