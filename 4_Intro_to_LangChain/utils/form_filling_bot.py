import gradio as gr
import random


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chains import LLMChain


class Key_Information_Company_Registry(BaseModel):
    """These are the list of key infromation about a Norwegian company that is registered as a public entity."""
    Organisasjonsnummer: int = Field("Organization number of the registered company.")
    Navn_foretaksnavn: str = Field("Name of the registered company.")
    Organisasjonsform: str = Field("Type of the organization.")
    Forretningsadresse : str = Field("The address of the business.")


def create_form_filling_chatbot():
    prompt_extraction = ChatPromptTemplate.from_messages([
        ("system", "Think carefully, and extract the info"),
        ("user", "{input}")
    ])
    
    prompt = ChatPromptTemplate.from_messages([
                ("system", "You are helpful but sassy assistant"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(temperature=0)
    
    llm_with_function = ChatOpenAI(temperature=0).bind(
        functions=[convert_pydantic_to_openai_function(Key_Information_Company_Registry)]
    )
    
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            )
        }
        | prompt
        | llm_with_function
        | OpenAIFunctionsAgentOutputParser()
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)

    chat_history_lc = []

    extraction_info = {}

    with gr.Blocks() as demo:
        with gr.Row():
            chatbot = gr.Chatbot()
            # with gr.Row():
            with gr.Column(scale=1):
                text_organisasjonsnummer = gr.Textbox(label = "Organisasjonsnummer")
                text_navn_foretaksnavn = gr.Textbox(label = "Navn_foretaksnavn")
                text_organisasjonsform = gr.Textbox(label = "Organisasjonsform")
                text_forretningsadresse = gr.Textbox(label = "Forretningsadresse")
        with gr.Row():
            msg = gr.Textbox(
                container=False,
                show_label=False,
                label="Message",
                placeholder="Type a message...",
                scale=7,
                autofocus=True,
            )
            submit_button = gr.Button(
                "Submit",
                variant="primary",
                scale=1,
                min_width=150,
            )
        clear = gr.ClearButton([msg, chatbot, text_organisasjonsnummer,
                                text_navn_foretaksnavn, text_organisasjonsform, text_forretningsadresse])
        def clear_history():
            # print("clear called")
            extraction_info.clear()
            chat_history_lc = []
            # print(extraction_info)
        
        clear.click(fn=clear_history)
        
        def respond(message, chat_history):
            
            for step in agent_executor.iter({"input": message}):
                # print(extraction_info)
                if output := step.get("intermediate_step"):
                    # print("here function call")
                    action, value = output[0]
                    extraction_info.update(action.tool_input) 
                    
                    bot_message = 'Form is now updated with new info'
                    break
                else:
                    bot_message = step['output']
                    # print(step['output'])
            chat_history.append((message, bot_message))
            # chat_history_lc.extend(
            #     [
            #         HumanMessage(content=message),
            #         AIMessage(content=bot_message),
            #     ]
            # )
            out_text_box = (
                out_collect('Organisasjonsnummer', extraction_info),
                out_collect('Navn_foretaksnavn', extraction_info),
                out_collect('Organisasjonsform', extraction_info),
                out_collect('Forretningsadresse', extraction_info),
            )
            out = ("", chat_history) 
            return out + out_text_box
    
        msg.submit(respond, [msg, chatbot], [msg, chatbot, text_organisasjonsnummer,
                                             text_navn_foretaksnavn, text_organisasjonsform,
                                             text_forretningsadresse])
        submit_button.click(respond, [msg, chatbot], [msg, chatbot, text_organisasjonsnummer,
                                             text_navn_foretaksnavn, text_organisasjonsform,
                                             text_forretningsadresse])
    return demo


def out_collect(key, dict):
    return dict[key] if key in dict else ""


