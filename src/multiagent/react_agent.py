from typing import Callable, Literal, Optional, Sequence, Type, TypeVar, Union, cast

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
)
from langchain_core.tools import BaseTool
from typing_extensions import Annotated, TypedDict

from langgraph._api.deprecation import deprecated_parameter
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.utils.runnable import RunnableCallable
import pdb

# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add


from pydantic import BaseModel

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    background: BaseMessage
    intermediate_output: str
    is_last_step: IsLastStep
    initial_prompt: BaseMessage



StateSchema = TypeVar("StateSchema", bound=AgentState)
StateSchemaType = Type[StateSchema]

STATE_MODIFIER_RUNNABLE_NAME = "StateModifier"

MessagesModifier = Union[
    SystemMessage,
    str,
    Callable[[Sequence[BaseMessage]], Sequence[BaseMessage]],
    Runnable[Sequence[BaseMessage], Sequence[BaseMessage]],
]

StateModifier = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], Sequence[BaseMessage]],
    Runnable[StateSchema, Sequence[BaseMessage]],
]


def _get_state_modifier_runnable(
    state_modifier: Optional[StateModifier], store: Optional[BaseStore] = None
) -> Runnable:
    state_modifier_runnable: Runnable
    if state_modifier is None:
        state_modifier_runnable = RunnableCallable(
            lambda state: state["messages"], name=STATE_MODIFIER_RUNNABLE_NAME
        )
    elif isinstance(state_modifier, str):
        _system_message: BaseMessage = SystemMessage(content=state_modifier)
        state_modifier_runnable = RunnableCallable(
            lambda state: [_system_message] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif isinstance(state_modifier, SystemMessage):
        state_modifier_runnable = RunnableCallable(
            lambda state: [state_modifier] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif callable(state_modifier):
        state_modifier_runnable = RunnableCallable(
            state_modifier,
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif isinstance(state_modifier, Runnable):
        state_modifier_runnable = state_modifier
    else:
        raise ValueError(
            f"Got unexpected type for `state_modifier`: {type(state_modifier)}"
        )

    return state_modifier_runnable


def _convert_messages_modifier_to_state_modifier(
    messages_modifier: MessagesModifier,
) -> StateModifier:
    state_modifier: StateModifier
    if isinstance(messages_modifier, (str, SystemMessage)):
        return messages_modifier
    elif callable(messages_modifier):

        def state_modifier(state: AgentState) -> Sequence[BaseMessage]:
            return messages_modifier(state["messages"])

        return state_modifier
    elif isinstance(messages_modifier, Runnable):
        state_modifier = (lambda state: state["messages"]) | messages_modifier
        return state_modifier
    raise ValueError(
        f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
    )


def _get_model_preprocessing_runnable(
    state_modifier: Optional[StateModifier],
    messages_modifier: Optional[MessagesModifier],
    store: Optional[BaseStore],
) -> Runnable:
    # Add the state or message modifier, if exists
    if state_modifier is not None and messages_modifier is not None:
        raise ValueError(
            "Expected value for either state_modifier or messages_modifier, got values for both"
        )

    if state_modifier is None and messages_modifier is not None:
        state_modifier = _convert_messages_modifier_to_state_modifier(messages_modifier)

    return _get_state_modifier_runnable(state_modifier, store)


def _should_bind_tools(model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools):
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
        )

    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI-style tool
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic-style tool
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            # unknown tool type so we'll ignore it
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")

    return False




from pydantic import BaseModel
import ast
from langchain_core.messages import HumanMessage

def create_react_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    response_schema: Optional[BaseModel],
    response_format,
    *,
    state_schema: Optional[StateSchemaType] = None,
    messages_modifier: Optional[MessagesModifier] = None,
    state_modifier: Optional[StateModifier] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
) -> CompiledGraph:

    if state_schema is not None:
        if missing_keys := {"messages", "is_last_step"} - set(
            state_schema.__annotations__
        ):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        tool_node = ToolNode(tools)
        # get the tool functions wrapped in a tool class from the ToolNode
        tool_classes = list(tool_node.tools_by_name.values())

    if _should_bind_tools(model, tool_classes):
        model = cast(BaseChatModel, model).bind_tools(tool_classes)

    # Define the function that determines whether to continue or not
    if response_schema is not None:
        def should_continue(state: AgentState) -> Literal["tools", "respond"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no function call, then we finish
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "respond"
                # return "__end__"
            # Otherwise if there is, we continue
            else:
                return "tools"
    else:
        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no function call, then we finish
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "__end__"
            # Otherwise if there is, we continue
            else:
                return "tools"

    # we're passing store here for validation
    preprocessor = _get_model_preprocessing_runnable(
        state_modifier, messages_modifier, store
    )
    

    if response_schema is not None:
        model_response_runnable = preprocessor | model.with_structured_output(response_schema, method='function_calling')
        def final_response(state: AgentState, config: RunnableConfig) -> AgentState:
            for _ in range(10):
                try:
                    result = model_response_runnable.invoke(state, config)
                    formatted_result = response_format(result)
                    
                    # Ensure complete state is returned, including intermediate_output
                    return {
                        **state,  # Preserve original state
                        **formatted_result  # Add formatted result
                    }
                except Exception as e:
                    print(f"⚠️ Error in final_response: {e}")
                    print("Error occured. Retrying...")
                    state["messages"].append("Error message: " + str(e))
            
            # If 10 retries all fail, return default state
            print("⚠️ Final response failed after 10 retries")
            return {
                **state,
                "intermediate_output": {}
            }


    model_runnable = preprocessor | model


    # Define the function that calls the model
    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        message_len = len(state["messages"])
        added_images = []

        for i, message in enumerate(reversed(state["messages"])):
            if isinstance(message, ToolMessage):

                if message.name == "image_to_base64":
                    try:
                        # Enhanced error handling for image_to_base64 tool
                        if isinstance(message.content, str):
                            import json
                            try:
                                message_content = json.loads(message.content)
                            except json.JSONDecodeError:
                                try:
                                    message_content = ast.literal_eval(message.content)
                                except (ValueError, SyntaxError):
                                    print(f"⚠️ Error parsing image_to_base64 content: {str(message.content)[:100]}...")
                                    continue
                        else:
                            message_content = message.content
                            
                        added_images.append(message_content)
                        state["messages"][message_len-i-1].content = "See the following human message for the image."
                    except Exception as e:
                        print(f"⚠️ Error processing image_to_base64: {e}")
                        continue
                elif message.name == "gemini_image_generator":
                    try:
                        # First try to parse as JSON if it's already a string representation of dict
                        if isinstance(message.content, str):
                            import json
                            try:
                                message_content = json.loads(message.content)
                            except json.JSONDecodeError:
                                try:
                                    message_content = ast.literal_eval(message.content)
                                except (ValueError, SyntaxError):
                                    print(f"⚠️ Error parsing gemini_image_generator content: {str(message.content)[:100]}...")
                                    continue
                        else:
                            # If it's already a dict, use it directly
                            message_content = message.content
                            
                        # Enhanced handling for different image keys that gemini_image_generator might return
                        if isinstance(message_content, dict):
                            image_processed = False
                            # Try different possible image keys in order of preference
                            for img_key in ["image_return", "image_path", "recent_image_path"]:
                                if img_key in message_content and message_content[img_key] is not None:
                                    image_content = message_content[img_key]
                                    
                                    # Handle different image content types
                                    if isinstance(image_content, str) and (image_content.startswith('outputs/') or image_content.endswith('.png') or image_content.endswith('.jpg')):
                                        # This is a file path, convert to base64 format
                                        try:
                                            import base64
                                            import os
                                            if os.path.exists(image_content):
                                                with open(image_content, 'rb') as img_file:
                                                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                                                    # Create proper OpenAI image format
                                                    image_obj = {
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": f"data:image/png;base64,{img_base64}"
                                                        }
                                                    }
                                                    added_images.append(image_obj)
                                                    print(f"📷 Converted gemini image path to base64: {image_content}")
                                            else:
                                                print(f"⚠️ Gemini image file not found: {image_content}")
                                                continue
                                        except Exception as e:
                                            print(f"⚠️ Error converting gemini image to base64: {e}")
                                            continue
                                    else:
                                        # Already in proper format (dict with image_url structure)
                                        added_images.append(image_content)
                                        print(f"📷 Using gemini image content directly: {type(image_content)}")
                                    
                                    # Mark image as processed and update message content
                                    message_content[img_key] = "See the following human message for the image."
                                    state["messages"][message_len-i-1].content = str(message_content)
                                    image_processed = True
                                    break
                            
                            if not image_processed:
                                print(f"⚠️ No valid image content found in gemini_image_generator response")
                                print(f"📝 Available keys: {list(message_content.keys())}")
                                print(f"📝 Content preview: {str(message_content)[:200]}...")
                        else:
                            # If content format is unexpected, just skip this message
                            print(f"⚠️ Unexpected content format from gemini_image_generator: {type(message_content)}")
                            print(f"📝 Content: {str(message_content)[:200]}...")
                            continue
                    except Exception as e:
                        print(f"⚠️ Error processing gemini_image_generator content: {e}")
                        print(f"📝 Content type: {type(message.content)}, Content: {str(message.content)[:200]}...")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    # Generic handling for other tools that might return complex content
                    try:
                        if isinstance(message.content, str) and (message.content.startswith('{') or message.content.startswith('[')):
                            # Looks like JSON or dict, try to parse it
                            import json
                            try:
                                message_content = json.loads(message.content)
                            except json.JSONDecodeError:
                                try:
                                    message_content = ast.literal_eval(message.content)
                                except (ValueError, SyntaxError):
                                    # If parsing fails, just use the string as-is
                                    print(f"⚠️ Could not parse {message.name} content, using as string")
                                    continue
                            
                            # Check if this tool also returns images
                            if isinstance(message_content, dict) and any(key in message_content for key in ["image", "image_return", "image_path"]):
                                print(f"🎯 Detected image content from tool: {message.name}")
                                for img_key in ["image", "image_return", "image_path"]:
                                    if img_key in message_content:
                                        # Handle different image content types
                                        image_content = message_content[img_key]
                                        
                                        # For file paths, create proper image message format
                                        if isinstance(image_content, str) and (image_content.startswith('outputs/') or image_content.endswith('.png') or image_content.endswith('.jpg')):
                                            # This is a file path, convert to base64 or proper format
                                            try:
                                                import base64
                                                import os
                                                if os.path.exists(image_content):
                                                    with open(image_content, 'rb') as img_file:
                                                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                                                        # Create proper OpenAI image format
                                                        image_obj = {
                                                            "type": "image_url",
                                                            "image_url": {
                                                                "url": f"data:image/png;base64,{img_base64}"
                                                            }
                                                        }
                                                        added_images.append(image_obj)
                                                        print(f"📷 Converted image path to base64 format: {image_content}")
                                                else:
                                                    print(f"⚠️ Image file not found: {image_content}")
                                                    continue
                                            except Exception as e:
                                                print(f"⚠️ Error converting image to base64: {e}")
                                                continue
                                        else:
                                            # Already in proper format or other format
                                            added_images.append(image_content)
                                            
                                        message_content[img_key] = "See the following human message for the image."
                                        state["messages"][message_len-i-1].content = str(message_content)
                                        break
                    except Exception as e:
                        print(f"⚠️ Error processing {message.name} content: {e}")
                        continue
                        
            elif len(added_images) > 0 and (isinstance(message, HumanMessage) or isinstance(message, AIMessage)) and not 'tool_calls' in message.additional_kwargs:
                for image in added_images:
                    state["messages"].insert(message_len-i, HumanMessage(
                        content=[image],
                    ))
                added_images = []
        # print(state)
        try:
            response = model_runnable.invoke(state, config)
        except Exception as e:
            print(e)
            for message in state["messages"]:
                if isinstance(message.content, list):
                    print("List of messages (Image URL)")
                else:
                    print(message)
            raise e
        
        if (
            state["is_last_step"]
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = await model_runnable.ainvoke(state, config)
        if (
            state["is_last_step"]
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableCallable(call_model, acall_model))
    workflow.add_node("tools", tool_node)

    if response_schema is not None:
        workflow.add_node("respond", final_response)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def route_tool_responses(state: AgentState) -> Literal["agent", "__end__"]:
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return "__end__"
        return "agent"

    if should_return_direct:
        workflow.add_conditional_edges("tools", route_tool_responses)
    else:
        workflow.add_edge("tools", "agent")
        if response_schema is not None:
            workflow.add_edge("respond", "__end__")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


# Keep for backwards compatibility
create_tool_calling_executor = create_react_agent

__all__ = [
    "create_react_agent",
    "create_tool_calling_executor",
    "AgentState",
]