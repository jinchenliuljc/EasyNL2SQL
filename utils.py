from langchain_community.utilities.sql_database import SQLDatabase
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
# from langchain.agents.format_scratchpad.openai_tools import (
#     format_to_openai_tool_messages,
# )
import dotenv
from typing import Dict, List
import sqlparse
from sqlparse.sql import Statement


dotenv.load_dotenv()



class ParamColumn(BaseModel):
    table_name: str = Field(description="the exact name of the table you want to explore.")
    calling_reason:str = Field(description="the reason why you think calling this tool is necessary")


class ParamTable(BaseModel):
    calling_reason:str = Field(description="the reason why you think calling this tool is necessary")


class ParamSubmit(BaseModel):
    SQL:str = Field(description="the sql you write")
    Knowledge:str = Field(description="""Summarize the knowledge you learned so far through the conversation with user and tool calling here in markdown format, make sure it is up to date and correct. **Include all the knowledge from your last summarization in the conversation, summarize them where neccessary and make adaptation based on what the user said**```e.g.
                           #knowledge about database
                           * Table A contains information about xxx
                           * Column aa is the xxx attribute of xxx
                           #knowledge from the user
                           * ABC stands for abc 
                           ```
                             """)
    Thought:str = Field(description = """Explain the logic of your SQL step by step, e.g.
                       ``` * step1: i select table A and join table B, step2: i filter data use xxxx, ... ```
                                        """)



class DBUtils:
    
    @classmethod
    def initialize(cls, db:SQLDatabase) -> None:
        cls.db= db

    @staticmethod
    @tool(args_schema=ParamTable)
    def get_table_names(calling_reason):
        '''Get all the table names and (sometimes) along with each a brief description'''
        # name = DBUtils.db.run(f"SELECT DATABASE();").strip("[](),'")
        # return DBUtils.db.run(f"SELECT table_name table_comment FROM information_schema.tables WHERE table_schema = '{name}';")
        return DBUtils.db.run("SELECT name FROM sqlite_master WHERE type = 'table';")
    
    @staticmethod
    @tool(args_schema=ParamColumn)
    def get_column_info(table_name, calling_reason):
        """Get the names, data_type, referenced_table_names, referenced_column_names of each columns of a table"""
        return DBUtils.db.run(f"SELECT sql FROM sqlite_master WHERE name = '{table_name}'")
        # name = DBUtils.db.run(f"SELECT DATABASE();").strip("[](),'")
        # rows = DBUtils.db.run(f'''SELECT 
        #     cols.column_name, 
        #     cols.data_type,
        #     kcu.referenced_table_name,
        #     kcu.referenced_column_name
        # FROM 
        #     information_schema.columns AS cols
        # LEFT JOIN 
        #     information_schema.key_column_usage AS kcu ON cols.table_schema = kcu.table_schema
        #     AND cols.table_name = kcu.table_name
        #     AND cols.column_name = kcu.column_name
        # WHERE 
        #     cols.table_schema = '{name}' 
        #     AND cols.table_name = '{table_name}';
        # ''')
        # column_info = '(column_name, data_type,referenced_table_name,referenced_column_name)'
        # answer = column_info + rows
        # return answer
    
    @staticmethod
    @tool(args_schema=ParamSubmit)
    def submit_sql(SQL, Knowledge, Thought):
        '''Submit the generated sql'''
        # print('sql:',sql)
        if DBUtils.is_select_statement(sql):
            try:
                DBUtils.db.run(sql)
                # return f"The sql '{sql}' is legitimate and error free, you can submit it."
                # return AgentFinish(return_values=sql, log='')
                
            except Exception as e:
                return f"An error occured when running the sql {e}"        
        else:
            return 'You can only write sql that is SELECT statement'
        
    
    @staticmethod
    def is_select_statement(sql):
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False
        statement = parsed[0]  # 取第一个语句
        if isinstance(statement, Statement) and statement.get_type() == 'SELECT':
            return True
        return False

