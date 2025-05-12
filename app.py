import autogen
import gradio as gr
import os
import time
from pathlib import Path
import multiprocessing as mp
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb 
import autogen
from autogen import AssistantAgent
from autogen.retrieve_utils import TEXT_FORMATS, get_file_from_url, is_url
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen.agentchat.contrib.retrieve_user_proxy_agent import (
    RetrieveUserProxyAgent,
    PROMPT_CODE,
)

TIMEOUT = 180
PROMPT_CODE = None
CHUNK_TOKEN_SIZE = 2000
DOCS_PATH = [os.path.join(os.path.dirname(__file__), "iBank_Product.docx"), os.path.join(os.path.dirname(__file__), "BankFAQs.csv")]
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])

CHROMA_DB_PATH="/app/tmp/chromadb"
COLLECTION_NAME = "customer_support_rag"
#vector_db = ChromaVectorDB(path=CHROMA_DB_PATH)
#chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
#collection = chroma_client.create_collection(name=COLLECTION_NAME)

config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_type": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": "AIzaSyC1bFgc22y4_I_18jbZxEgi4iJYDVaf5Wc",
    },
]

llm_config={
    #"request_timeout": TIMEOUT,
    #"seed": 42,
    "config_list": config_list,
    "temperature": 0
}

model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="AIzaSyC1bFgc22y4_I_18jbZxEgi4iJYDVaf5Wc",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
    },
)

custom_css = """
           
            .message-row img {
                margin: 0px !important;
            }

            .avatar-container img {
            padding: 0px !important;
}
        """

# Initialize Agents
def initialize_agents(llm_config, docs_path=None):
    assistant = AssistantAgent(
        name="assistant",
        system_message= '''
        You are Luna, a customer service staff at a bank. Reply customer questions in Vietnamese. 
        
        You're helpful, professional, clever, and friendly, and your main job is to assist users with anything related to banking. 
        You’re an expert in all things about banking, offering answers and guidance based on the company's knowledge base.
        Only use bullet points to answer question.

        Rules for Chatbot Behavior:
        Only respond using the given knowledge base.
        Provide clear, confident, and polite answers every time.
        Keep responses aligned with the company’s tone and style.
        Avoid giving information outside of scope.
        Avoid mentioning human agent unless question is out of scope.
        If question is out of scope, say you will connect users to human agent.
        ''',
        #model_client=model_client,
        llm_config=llm_config,
        max_consecutive_auto_reply=2
    )

    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "docs_path": docs_path,
            'embedding_model': 'all-MiniLM-L6-v2',
            #"vector_db": vector_db,
            #"overwrite": False,
            "get_or_create": True,
            #"chunk_token_size": CHUNK_TOKEN_SIZE,
            #"customized_prompt": PROMPT_CODE,
            "custom_text_split_function": text_splitter.split_text,
            "collection_name": COLLECTION_NAME 
        },
        code_execution_config={"use_docker":False}
    )

    return assistant, ragproxyagent

# Initialize Chat
def initiate_chat(config_list, problem, queue, n_results=3):
    global assistant, ragproxyagent
    assistant.reset()
    try:
        ragproxyagent.initiate_chat(
            assistant, message=ragproxyagent.message_generator, problem=problem, #, n_results=n_results
        )
        messages = ragproxyagent.chat_messages
        messages = [messages[k] for k in messages.keys()][0]
        messages = [m["content"] for m in messages if m["role"] == "user"]
    except Exception as e:
        messages = [str(e)]
    queue.put(messages)

# Wrap AutoGen part into a function
def chatbot_reply(input_text):
    """Chat with the agent through terminal."""
    queue = mp.Queue()
    process = mp.Process(
        target=initiate_chat,
        args=(config_list, input_text, queue),
    )
    process.start()
    try:
        messages = queue.get(timeout=TIMEOUT)
    except Exception as e:
        messages = [str(e) if len(str(e)) > 0 else "Invalid Request to Gemini, please check your API keys."]
    finally:
        try:
            process.terminate()
        except:
            pass
    return messages

global assistant, ragproxyagent

# Set up UI with Gradio
with gr.Blocks(css = custom_css) as demo:
    assistant, ragproxyagent = initialize_agents(llm_config, docs_path=DOCS_PATH)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "autogen.jpg"))),
        type='messages'
        # height=600,
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
    )

    clear = gr.ClearButton([txt_input, chatbot])

    def user(user_message, chat_history):
        chat_history.append({"role": "user", "content": user_message})
        return "", chat_history
    
    def respond(chat_history):
        #chat_history.append({"role": "user", "content": user_message})
        bot_message = chatbot_reply(chat_history[-1]['content'])
        print(bot_message)
        bot_message_res = (
            bot_message[-1]
            if len(bot_message) > 0 and bot_message[-1] != "TERMINATE"
            else bot_message[-2]
            if len(bot_message) > 1 
            else "Tôi không thể trả lời câu hỏi này. Tôi sẽ kết nối bạn với nhân viên hỗ trợ."
        )
        print(bot_message)
        chat_history.append({"role": "assistant", "content": bot_message_res})
        return chat_history
    
    #def respond(user_message, chat_history):
    #    #chat_history.append({"role": "user", "content": user_message})
    #    bot_message = chatbot_reply(user_message)
    #    bot_message_res = (
    #        bot_message[-1]
    #        if len(bot_message) > 0 and bot_message[-1] != "TERMINATE"
    #        else bot_message[-2]
    #        if len(bot_message) > 1
    #        else "Context is not enough for answering the question. Please press `enter` in the context url textbox to make sure the context is activated for the chat."
    #    )
    #    chat_history.append({"role": "assistant", "content": bot_message_res})
    #    return "", chat_history
    
    # Params for submit(): function, function input, function output
    txt_input.submit(user, [txt_input, chatbot], [txt_input, chatbot], queue=False).then( 
            respond,
            [chatbot], 
            [chatbot], 
            queue=False
    )
    #txt_input.submit(respond, [txt_input, chatbot], [txt_input, chatbot], queue=False)

if __name__ == "__main__":
    demo.launch(share=True)
