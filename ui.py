import chainlit as cl

import os
from dotenv import load_dotenv
load_dotenv()

from collections.abc import Callable, Coroutine
from typing import Any

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import AutoFunctionInvocationContext

from plugins.proverbs_plugin import *

# List of kernel functions to disable showing as a step in the chainlit
kernel_functions_to_exclude = ["chat", "extract_actions"]

kernel = Kernel()

async def init():
    await init_kernel()
    init_function_call_handlers()

async def init_kernel():
    global kernel, previous_message_id, chat_history, chat_function, extract_actions_function, previous_message

    kernel = Kernel() 
    system_message = get_system_prompt()

    service_id = "assistant-chat"

    if os.getenv("SERVICE_TYPE") == "openai":
        chat_service = OpenAIChatCompletion(
            service_id=service_id, env_file_path=".env" )
    else:
        chat_service = AzureChatCompletion(
            service_id=service_id, env_file_path=".env" )

    kernel.add_service(chat_service)

    req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    req_settings.max_tokens = 4096
    req_settings.temperature = 0.0
    req_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        filters={"excluded_plugins": ["chat"]}
    )
    #req_settings.parallel_tool_calls = False

    chat_function = kernel.add_function(
        prompt=system_message + """{{$chat_history}}{{$user_input}}""",
        function_name="chat",
        plugin_name="chat",
        description="Main chat function",
        prompt_execution_settings=req_settings,
    )    

    req_settings_extract = kernel.get_prompt_execution_settings_from_service_id("assistant-chat")
    req_settings_extract.max_tokens = 4096
    req_settings_extract.temperature = 0.1

    prompt_template = """Extract the possible actions (short button texts) from the LLM response below. 
            {{$llmResponse}}
            The actions should have the following JSON format. Only return the JSON, don't use a json block, just the pure json string
            [
                {
                    "text": "Action 1",
                    "value": "action1"
                },
                {
                    "text": "Action 2",
                    "value": "action2"
                }
            ]
            The actions should be extracted from the LLM provided possible responses, e.g. the missing words or meanings of proverbs.
            Only extract the actions that are relevant to the user's input and are related to possible actions that the user can take.
            Use the exact words from the response to fill the value field.
            The value is the short text that is used in the action button. """    

    extract_actions_function = kernel.add_function(
        function_name="extract_actions",
        plugin_name="utility",
        prompt=prompt_template,
        prompt_execution_settings=req_settings_extract
    )
    
    previous_message_id = None
    previous_message = None
    chat_history = ChatHistory(system_message=system_message)   

    await init_plugins()

def get_system_prompt():

    # Read the content of the prompt file
    with open(os.path.join('prompts', '00-proverb_ai_system_prompt.txt'), 'r') as file:
        system_prompt = file.read()

    system_prompt = system_prompt + """If tool paramteters are not specified, always ask the user for the parameters before calling the tool. 
                    Don't assume parameters of tools if not provided earlier."""
    
    return system_prompt

async def init_plugins():
    global kernel, proverb_plugin

    proverb_plugin = await ProverbsPlugin.create(kernel)
    kernel.add_plugin(proverb_plugin, plugin_name="proverbplugin")


async def function_call_filter(
    context: AutoFunctionInvocationContext,
    next: Callable[[AutoFunctionInvocationContext], Coroutine[Any, Any, None]],
) -> None:

    if context.function.name in kernel_functions_to_exclude and context.function.name != "create_plan":
        await next(context)
        return
    
    await next(context)

    function_param_keys = ", ".join([f"{k.name}" for k in context.function.metadata.parameters])
    arguments_str = "\n".join([f"**{k}**: *'{v}'*" for k, v in context.arguments.items() if k in function_param_keys])  

    result = context.result

    tool_call_str = f"Function called: {context.function.name}({arguments_str}), result: {result.value}"
    chat_history.add_assistant_message(tool_call_str)

    add_chainlit_step(context.function.name, arguments_str, result.value)

def init_function_call_handlers():
    global kernel
    kernel.add_filter("function_invocation", function_call_filter)
    
async def get_ai_response(user_input: str) -> str:
    global kernel, chat_function, chat_history
    return kernel.invoke_stream(
        chat_function,
        user_input=user_input,
        chat_history=chat_history,
    )

def add_to_chat_history(user_input: str, assistant_response: str):
    global chat_history
    chat_history.add_user_message(user_input)
    chat_history.add_assistant_message(assistant_response)

# --- Chainlit logic from here ---

async def extract_actions_from_response(response: str) -> list[cl.Action]:
    global extract_actions_function

    actions_json_func_result = await kernel.invoke(
        extract_actions_function,
        llmResponse=response,
    )
    print(actions_json_func_result)

    try:
        actions_json = json.loads(str(actions_json_func_result))
    except:
        return

    chainlit_actions = [
        cl.Action(name="response_action", value=action["value"], label=action["text"])
        for action in actions_json
    ]   

    return chainlit_actions

@cl.on_chat_start
async def main():
    await init()

@cl.action_callback("response_action")
async def on_action(action: cl.Action):
    if (previous_message is not None):
        await previous_message.remove_actions()
    msg = cl.Message(type="user_message", content=action.label)
    await msg.send()
    await on_message(msg)
    return action.value

@cl.on_message  
async def on_message(message: cl.Message):
    global previous_message_id, proverb_plugin, previous_message
    user_input = message.content
   
    msg = cl.Message(content="")
    await msg.send()
    previous_message_id = msg.id
    previous_message = msg

    answer = await get_ai_response(user_input) 

    async for message in answer:
        if token := str(message[0]) or "":
            await msg.stream_token(token)
    await msg.update()

    answer_str = str(msg.content)

    # Extract actions from the response
    actions = await extract_actions_from_response(answer_str)
    if actions:
        msg.actions = actions
        await msg.update()

    add_to_chat_history(user_input, answer_str)

def add_chainlit_step(name: str, step_text: str, func_result: str = None):
    global previous_message_id
    
    with cl.Step(name=name, type="tool", parent_id=previous_message_id) as step:
        step.output = step_text
        # if func_result:
        #     step.output += f"\n\n**Tool use result**: {func_result}"

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Véletlenszerű közmondás",
            message="Adj nekem 5 db véletlenszerű közmondást, magyarázattal",
            icon="/public/random.svg",
            ),                    
        cl.Starter(
            label="Magyarázz el egy közmondást",
            message="Magyarázd el nekem a közmondást.",
            icon="/public/explain.svg",
            ),
        cl.Starter(
            label="Játék: hiányzó szó kitalálása",
            message="Hiányzó szó kitalálós játék",
            icon="/public/missing_word.svg",
            ),
        cl.Starter(
            label="Játék: közmondás jelentésének kitalálása",
            message="Közmondás jelentésének kitalálós játék",
            icon="/public/meaning.svg",
            ),                                                                  
        ]

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)        