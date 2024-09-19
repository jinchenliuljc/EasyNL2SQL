import openai
import dotenv

from openai import OpenAI


dotenv.load_dotenv()
client = OpenAI()



# class SummaryAgent():
#     pass



def summary_agent(history):
    system_prompt = """You are a experienced data analyst AI who generate SQL to extract data from database based on the user's request. 
    Below you will see a conversation you had between you and the user. Here is what you need to do.
    # Requirements
    *  Based on the conversation, summary the user's request on data in two sentences. The summary will be used to retrieve all knowledge related to this conversation next time similar request comes up. You should try to retain all feature that could identify this conversation from others, e.g. terms/ abbreviations. Example:```{conversation:{"Human" : "I want to see sales of AAA in year 2020", "AI": "SQL:XXXX", "Human,"Actually I only want to see Dec", "AI":"SQL:XXXX","Human":"Show me only the first few rows", "AI":"SQL:XXX"},Your output: "The user wants to see the sales of AAA in December 2020, and only the first few lines"  } ```
    """+f"""
    # Conversation
    ```
    {history}
    ```
    """+"""
    # Output format
    Output in this json format and NOTHING ELSE
    {"feature_identified":"["aaa means xxx","bbb means yyy"]","summary":"YOUR SUMMARY that includes feature_identified"}
    """
    print(system_prompt)
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": system_prompt},
    ],
    response_format= { "type": "json_object" },
    temperature=0
    ) 
    return response.choices[0].message.content


def merge_agent():
    pass
