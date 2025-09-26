import sys
import os
# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START


from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal, Sequence, Dict, Any
from typing_extensions import TypedDict
import operator
import ast
import traceback
import functools
import json
import pdb
from react_agent import create_react_agent
import tools as tools
import colors as colors

class AgentTeam():

    def __init__(self, config, llm, member_list=[], member_names=[], member_info=[], team_name="Default", supervisor_name="FINISH",
                 intermediate_output_desc="", final_output_form="", fullshared_memory=False):
        self.config = config
        self.llm = llm
        self.graph = None

        self.member_list = member_list
        self.member_info = member_info
        self.member_names = member_names
        self.team_name = team_name
        self.sup_name = supervisor_name
        
        self.intermediate_output_desc = intermediate_output_desc
        self.final_output_form = final_output_form
        self.fullshared_memory = fullshared_memory

        self.message_prompt = f"If the next agent is one of {', '.join(member_names)}, give detailed instructions and requests. If {supervisor_name}, report a summary of all results."
        self.thought_prompt = "Output a detailed analysis on the most recent message. In detail, state what you think should be done next, and who you should contact next."
    


    def loadSuperviser(self, additional_prompt=""):
        options = [self.sup_name] + self.member_names
        job_list = "\n".join([mem + ": " + pr for mem, pr in zip(self.member_names, self.member_info)])
        prompt_list = [
                ("system",
                 "You are a supervisor tasked with managing a conversation between the following workers: {members}. "
                 "Given the following messages, respond with the worker to act next."
                 "Each worker will perform a task and respond with their results and status."
                 "\nJoblist: \n{joblist}"),
                
                ("system", "The current background is: {background}"),
                
                ("system", "Conversation History:"),
                
                MessagesPlaceholder(variable_name="messages", n_messages=20),
                
                ("system",
                 "Given the conversation above, output the following in this exact order:\n"
                 "1. 'thoughts': {thought_prompt}\n"
                 "2. 'messages': {message_prompt}\n"
                 "3. Who should act next? Select one of: {options} and output as 'next'."
                 " When you have determined that the final output is gained, report back with {finish} as 'next'.\n"
                 "4. The detailed background of the problem you are trying to solve (given in the first message) as 'background'.\n"
                 "5. The intermediate outputs to give as 'intermediate_output'.\n"
                 "" + additional_prompt + ""
                 "{error}"),
            ]
        if self.team_name == "Default":
            prompt_list.pop(1)
        
        prompt = ChatPromptTemplate.from_messages(prompt_list).partial(options=str(options), members=", ".join(self.member_names), joblist=job_list,\
                  finish=self.sup_name, sup_name=self.team_name + " Supervisor",\
                  thought_prompt=self.thought_prompt, message_prompt=self.message_prompt)



        class routeResponse(BaseModel):
            thoughts: str = Field(description=self.thought_prompt)
            next: Literal[(*options, )]
            messages: str = Field(description=self.message_prompt)
            intermediate_output: str = Field(description=self.intermediate_output_desc)
            background: str
                
            @field_validator('messages', 'intermediate_output', mode="before")
            def cast_to_string(cls, v):
                return str(v)
            
        
        def supervisor_agent(state):
            prev_history = state["history"]

            if self.fullshared_memory:
                state["messages"] = list(state["history"]["all"])
            else:
                state["messages"] = list(state["history"][self.team_name + " Supervisor"])

            state["error"] = ""
            state["messages"] = state["messages"][-15:]
            state["history"] = {}
            
            # Check for zero-revision mode
            zero_revision_mode = state.get("zero_revision_mode", False)
            skip_evaluation_cycles = state.get("skip_evaluation_cycles", False)
            
            # Include recent_files information in the state for context
            recent_files_info = ""
            if "recent_files" in state and state["recent_files"]:
                recent_files_info = "\n\nAvailable Files:\n"
                for key, path in state["recent_files"].items():
                    if path and isinstance(path, str):
                        recent_files_info += f"- {key}: {path}\n"
                        # Special handling for image evaluators
                        if "most_recent_image_filepath" in key or "recent_image" in key:
                            recent_files_info += f"  (Use this path for image evaluation tools)\n"

            # If this is the main team, there is no background.
            if self.team_name == "Default":
                state["background"] = ""
            else:
                background_content = state["background"].content
                
                # 🔧 FIX: Remove any accumulated "The current background is:" prefixes
                if background_content.startswith("The current background is:"):
                    # Extract the original content after the prefix
                    lines = background_content.split('\n')
                    first_line = lines[0]
                    if first_line.startswith("The current background is:"):
                        # Remove the prefix and get the actual content
                        actual_content = first_line[len("The current background is:"):].strip()
                        if len(lines) > 1:
                            # Rejoin with remaining lines
                            background_content = actual_content + '\n' + '\n'.join(lines[1:])
                        else:
                            background_content = actual_content
                
                # Append file information to background
                if recent_files_info:
                    background_content += recent_files_info
                
                # Add zero-revision mode information to background
                if zero_revision_mode:
                    background_content += "\n\n🚀 ZERO-REVISION MODE: max_revisions_per_team = 0"
                    background_content += "\n📝 Skip all GraphicRevisor activities and proceed directly to completion after ContentCreation."
                    background_content += "\n⚡ Goal: Generate initial banner and evaluate, then finish (no revisions needed)."
                
                state["background"] = background_content

            supervisor_chain = prompt | self.llm.with_structured_output(routeResponse, method='function_calling')

            for attempt in range(10):
                try:
                    result = supervisor_chain.invoke(state)
                    
                    # Zero-revision mode logic: override next agent selection
                    if zero_revision_mode and result.next == "GraphicRevisor":
                        print(f"🚀 [Zero-Revision] Skipping GraphicRevisor, proceeding to finish")
                        result.next = self.sup_name  # Finish the team
                        result.messages = "Zero-revision mode: Initial banner completed, proceeding to Judge evaluation without revisions."
                    
                    # Skip redundant evaluation cycles in zero-revision mode
                    if zero_revision_mode and skip_evaluation_cycles and result.next in ["EvaluationTeam", "TextContentEvaluator", "LayoutEvaluator", "BackgroundImageEvaluator"]:
                        # Allow only one evaluation cycle, then finish
                        evaluation_count = sum(1 for msg in prev_history.get(self.team_name + " Supervisor", []) if "evaluation" in str(msg.content).lower())
                        if evaluation_count >= 1:
                            print(f"🚀 [Zero-Revision] Skipping additional evaluation cycles, proceeding to finish")
                            result.next = self.sup_name  # Finish the team
                            result.messages = "Zero-revision mode: Initial evaluation completed, proceeding to Judge evaluation."
                    
                    if len(state["messages"]) > 1:
                        # print(state["messages"][-2].content)
                        # print(result.messages)
                        if result.messages in state["messages"][-2].content:
                            print("Same message as the previous message. Retrying...")
                            state["error"] += "\n\nERROR: DO NOT generate the same message again. If you are not sure what to do next, make sure to re-think what you need to do in 'thoughts'."
                            # result.messages = "Please restate clearly what you want me to do next."
                            continue
                    break
                except Exception as e:
                    print(e)
                    print("Error occured. Retrying...")
                    state["error"] = "\n\nDouble check that 'next' is one of:  {options}."
            
            # Enhance messages for TextContentEvaluator with specific image path
            enhanced_messages = result.messages
            if result.next == "TextContentEvaluator" and "recent_files" in state and state["recent_files"]:
                # Find the most recent image path
                image_path = None
                for key, path in state["recent_files"].items():
                    if ("most_recent_image_filepath" in key or "recent_image" in key) and path:
                        image_path = path
                        break
                
                if image_path:
                    enhanced_messages += f"\n\nIMPORTANT: Use this exact image path for evaluation: {image_path}"
                    enhanced_messages += f"\nCall the image_to_base64 tool with path parameter: {image_path}"
            
            # Enhance messages for GraphicRevisor with specific image path for revision
            elif result.next == "GraphicRevisor" and "recent_files" in state and state["recent_files"]:
                # Find the most recent image path for GraphicRevisor to revise
                image_path = None
                for key, path in state["recent_files"].items():
                    if ("most_recent_image_filepath" in key or "recent_image" in key) and path:
                        image_path = path
                        break
                
                if image_path:
                    # 🎯 DUAL INPUT: Find logo file in output directory for dual input
                    import glob, os
                    output_folder = None
                    try:
                        # Try to get output folder from state or from image_path directory
                        if os.path.exists(image_path):
                            output_folder = os.path.dirname(image_path)
                    except:
                        pass
                    
                    logo_path = None
                    if output_folder:
                        logo_files = glob.glob(f"{output_folder}/*.png")
                        print(f"🔍 [Supervisor] Output folder: {output_folder}")
                        print(f"🔍 [Supervisor] Found PNG files: {[os.path.basename(f) for f in logo_files]}")
                        for file in logo_files:
                            if not os.path.basename(file).startswith('generated_image_'):
                                logo_path = file
                                print(f"🎯 [Supervisor] Found logo file: {logo_path}")
                                break
                        if not logo_path:
                            print(f"⚠️ [Supervisor] No logo file found in {output_folder}")
                    if logo_path and os.path.exists(logo_path):
                        # Dual input: previous banner + logo
                        enhanced_messages += f"\n\n🎯 DUAL INPUT REQUIREMENT: Use BOTH images for revision"
                        enhanced_messages += f"\n📷 Previous banner: {image_path}"
                        enhanced_messages += f"\n🏷️ Original logo: {logo_path}"
                        enhanced_messages += f"\n📝 Call gemini_image_generator with input_filepath parameter: ['{image_path}', '{logo_path}']"
                        enhanced_messages += f"\n⚠️ CRITICAL: Use EXACT paths above, include both images for logo consistency"
                    else:
                        # Single input: just previous banner
                        enhanced_messages += f"\n\n🎯 IMPORTANT: Use this exact image path for revision: {image_path}"
                        enhanced_messages += f"\n📝 Call the gemini_image_generator tool with input_filepath parameter: ['{image_path}']"
                        enhanced_messages += f"\n⚠️ Make sure to use the exact path format: ['{image_path}'] in square brackets as a list"
            
            new_msg = AIMessage(content=enhanced_messages, name=self.team_name + "_Supervisor")
            if not ("Intermediate Output" in new_msg.content or "Final Output" in new_msg.content):
                if result.intermediate_output in ["", "{{}}"]:
                    result.intermediate_output = state["intermediate_output"]
                if not str(result.intermediate_output) in new_msg.content:
                    new_msg.content = new_msg.content + "\n\nFinal Output: " + str(result.intermediate_output)

            if isinstance(result.intermediate_output, str):
                try:
                    result.intermediate_output = ast.literal_eval(result.intermediate_output)
                except Exception as e:
                    try:
                        result.intermediate_output = json.loads(result.intermediate_output)
                    except Exception as e2:
                        print("Could not parse.")# Current output:", result.intermediate_output)
                        result.intermediate_output = state["intermediate_output"]
                        print(e)
                        print(e2)
                if result.intermediate_output == {}:
                    result.intermediate_output = state["intermediate_output"]
            
            if self.fullshared_memory:
                new_history = {
                        **prev_history,
                        self.team_name + " Supervisor": prev_history.get(self.team_name + " Supervisor", []) + [new_msg.model_copy()],
                        result.next: prev_history.get(result.next, []) + [new_msg.model_copy()],
                        "all": prev_history.get("all", []) + [new_msg.model_copy()]
                    }
            else:
                new_history = {
                        **prev_history,
                        self.team_name + " Supervisor": prev_history.get(self.team_name + " Supervisor", []) + [new_msg.model_copy()],
                        result.next: prev_history.get(result.next, []) + [new_msg.model_copy()]
                    }
            
            new_history[self.team_name + " Supervisor"][-1].content = "{Thoughts: " + result.thoughts + "}\n\n" + new_history[self.team_name + " Supervisor"][-1].content
                    

            return {
                "intermediate_output": result.intermediate_output,
                "messages": new_msg,
                "background": AIMessage(content=result.background, name=self.team_name + " Supervisor"),
                "next": result.next,
                "history": new_history
            }
        

        return supervisor_agent
    

    def createStateGraph(self, additional_prompt=""):
            class AgentState(TypedDict):
                # history: Annotated[Sequence[BaseMessage], operator.add]
                history: Dict[str, Annotated[Sequence[BaseMessage], operator.add]]
                messages: BaseMessage
                background: BaseMessage
                intermediate_output: Dict[str, Any] = Field(description=self.intermediate_output_desc)
                recent_files: Dict[str, str] = Field(default_factory=dict, description="Tracks recently generated file paths by agent/tool")
                next: str
                initial_prompt: BaseMessage

            if self.llm is None:
                self.loadLLM()
            workflow = StateGraph(AgentState)

            for member, member_name in zip(self.member_list, self.member_names):
                workflow.add_node(member_name, member)
            
            workflow.add_node(self.team_name + " Supervisor", self.loadSuperviser(additional_prompt=additional_prompt))
            for member in self.member_names:
                workflow.add_edge(member, self.team_name + " Supervisor")
            
            
            
            conditional_map = {k: k for k in self.member_names}
            conditional_map[self.sup_name] = END
            workflow.add_conditional_edges(self.team_name + " Supervisor", lambda x: x["next"], conditional_map)
            # Finally, add entrypoint
            workflow.add_edge(START, self.team_name + " Supervisor")

            self.graph = workflow.compile(debug = False)
            return functools.partial(AgentTeam.prompt, graph=self.graph, team_name=self.team_name, sup_name=self.sup_name if self.sup_name != "FINISH" else "", fullshared_memory=self.fullshared_memory)


    def prompt(state, config, graph, team_name, sup_name, fullshared_memory):
        int_output = {}
        color_assign = colors.TerminalColor()
        prev_len = len(state["history"][team_name + " Supervisor"])

        if "initial_prompt" in state:
            prev_initial_prompt = state["initial_prompt"]
            if not fullshared_memory:
                state["initial_prompt"] = "Prompter: " + state["history"][team_name + " Supervisor"][-1].name + "\n" + state["history"][team_name + " Supervisor"][-1].content
        else:
            prev_initial_prompt = state["history"][team_name + " Supervisor"][-1].content
            state["initial_prompt"] = prev_initial_prompt

        for s in graph.stream(
            state, config
        ):
            if "__end__" not in s:
                key = list(s.keys())[0]
                if key in ['history', 'messages', 'background', 'intermediate_output', 'next']:
                    result = s
                    try:
                        key = result['background'].name
                    except Exception as e:
                        print(e)
                        print(s["background"])
                        assert "Background error"
                    if prev_len == len(result["history"][team_name + " Supervisor"]):
                        continue
                else:
                    result = s[key]

                if int_output == {} and result["intermediate_output"] != {}:
                    int_output = result["intermediate_output"]
                
                team_id_display = ""
                if "team_id" in state:
                    team_id_display = f" Team{state['team_id']}"
                elif hasattr(state, 'get') and state.get('team_id'):
                    team_id_display = f" Team{state.get('team_id')}"
                
                print("Agent:", color_assign.colorText(key + team_id_display + f" ({len(result['history'][key])})", key))
                if key == team_name + " Supervisor":
                    print("Messages:")
                    print(color_assign.colorText(result["history"][key][-1].pretty_repr(), key))
                    print()

                if result["next"] != sup_name:
                    print("Background:")
                    print(color_assign.colorText(str(result["background"].content), key))
                    print()
                    if key != result["next"]:
                        print(color_assign.colorText(key, key), "->", color_assign.colorText(result["next"], result["next"]))
                    else:
                        print(color_assign.colorText(key, key), "->", color_assign.colorText(team_name + " Supervisor", team_name + " Supervisor"))
                    print()
        # print(s)
        state["initial_prompt"] = prev_initial_prompt
        if sup_name == "":
            return [result, int_output]
        return result
    


class ReactAgent():

    def __init__(self, config, intermediate_output_desc="", llm=None, key_type="GPT4", fullshared_memory=False):
        self.intermediate_output_desc = intermediate_output_desc
        self.config = config
        self.llm = llm

        self.key_type = key_type
        self.fullshared_memory = fullshared_memory

    def responseFormatter(result) -> dict:
        return {
                "intermediate_output": result.intermediate_output
            }


    def loadMember(self, name, sel_tools, member_prompt, sup_name):
        # Create unique agent name if team_id is available (for competitive workflow)
        unique_agent_name = name
        if hasattr(self, 'team_id') and self.team_id:
            unique_agent_name = f"{name}_Team{self.team_id}"
            print(f"🎯 [Team {self.team_id}] Creating unique agent: {unique_agent_name}")
        
        agent_tools = tools.getTools(sel_tools, self.config, self.llm, unique_agent_name)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", member_prompt),
                ("system", "Background Information: {background}"),
                # ("system", "Incoming Instructions:\n{initial_prompt}"),
                ("system", "Conversation History:"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        class routeResponse(BaseModel):
            intermediate_output: str = Field(description=self.intermediate_output_desc)
                
            # @field_validator('intermediate_output', mode="before")
            # def cast_to_string(cls, v):
            #     return str(v)

        agent = create_react_agent(self.llm, tools=agent_tools, debug=False, state_modifier=prompt, response_schema=routeResponse, \
                                   response_format=ReactAgent.responseFormatter)
        return functools.partial(ReactAgent.agent_node, agent=agent, name=name, sup_name=sup_name, fullshared_memory=self.fullshared_memory)
    
    
    
    def agent_node(state, agent, name, sup_name="", fullshared_memory=False):
        color_assign = colors.TerminalColor()

        if fullshared_memory:
            state["messages"] = state["history"]["all"][-15:]
        else:
            state["messages"] = state["history"][name][-15:]
        prev_history = state["history"]
        state["history"] = {}

        prev_initial_prompt = state["initial_prompt"]
        
        if not fullshared_memory:
            # Safely handle message name that might be None
            last_message = prev_history[name][-1]
            message_name = last_message.name if last_message.name is not None else "Unknown"
            message_content = last_message.content if last_message.content is not None else ""
            state["initial_prompt"] = "Prompter: " + message_name + "\n" + message_content
        

        prev_len = len(state["messages"])
        result = None  # Initialize result variable
        captured_intermediate_output = {}  # Capture intermediate_output

        for result in agent.stream(state, {}):
            if "__end__" not in result:
                
                # Capture intermediate_output (may appear in different steps)
                if "intermediate_output" in result:
                    captured_intermediate_output = result["intermediate_output"]
                    print(f"🔍 Captured intermediate_output: {type(captured_intermediate_output)}")
                
                # Safe check for messages key before accessing
                if "messages" in result and len(result["messages"]) != prev_len:
                    resp = result["messages"][-1]
                    if isinstance(resp, ToolMessage) and (resp.name == "output_tool" or resp.name == "image_to_base64" or resp.name == "gemini_image_generator"):
                        continue
                    
                    # Extract file paths from tool messages
                    if isinstance(resp, ToolMessage):
                        if resp.name in ["recraft_image_generator", "gemini_image_generator"]:
                            try:
                                if isinstance(resp.content, dict):
                                    tool_result = resp.content
                                elif isinstance(resp.content, str):
                                    # Try to parse string content as JSON
                                    try:
                                        tool_result = json.loads(resp.content)
                                    except json.JSONDecodeError:
                                        # If JSON parsing fails, try ast.literal_eval
                                        try:
                                            tool_result = ast.literal_eval(resp.content)
                                        except (ValueError, SyntaxError):
                                            # If both fail, log the content and skip
                                            print(f"⚠️ Could not parse tool response content: {str(resp.content)[:100]}...")
                                            continue
                                else:
                                    # Skip processing if content type is unexpected
                                    print(f"⚠️ Unexpected tool response content type: {type(resp.content)}")
                                    continue
                                
                                if isinstance(tool_result, dict) and "recent_image_path" in tool_result:
                                    # Update recent_files in the state
                                    if "recent_files" not in state:
                                        state["recent_files"] = {}
                                    state["recent_files"][f"{name}_recent_image"] = tool_result["recent_image_path"]
                                    state["recent_files"]["most_recent_image_filepath"] = tool_result["recent_image_path"]
                                    print(f"📁 Updated recent_files: {name} -> {tool_result['recent_image_path']}")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not extract file path from tool result: {e}")
                                # Print more details for debugging
                                print(f"📝 Tool name: {resp.name}, Content type: {type(resp.content)}")
                                if isinstance(resp.content, str):
                                    print(f"📝 Content preview: {resp.content[:200]}...")
                                continue
                    
                    # print(result)
                    try:
                        team_id_display = ""
                        if "team_id" in state:
                            team_id_display = f" Team{state['team_id']}"
                        elif hasattr(state, 'get') and state.get('team_id'):
                            team_id_display = f" Team{state.get('team_id')}"
                        print("Agent:", color_assign.colorText(name + team_id_display + f' ({len(result["messages"])})', name))
                    except (ValueError, IOError) as e:
                        # Handle closed file error silently or log to a file if needed
                        print(e)
                        pass
                    print(color_assign.colorText(resp.pretty_repr(), name))
                    print()
                    prev_len = len(result["messages"])
                elif "messages" not in result:
                    # Debug log for missing messages key
                    print(f"🔍 Debug: result missing 'messages' key. Available keys: {list(result.keys())}")

        # Ensure result is not None and contains necessary keys
        if result is None:
            print(f"⚠️ Warning: No result generated for agent {name}")
            result = {
                "messages": [],
                "intermediate_output": {},
                "background": None
            }

        # Safe access to result messages after the loop
        if "messages" in result and len(result["messages"]) > 0:
            new_msg = AIMessage(content=result["messages"][-1].content, name=name.replace(" ", "_"))
            # Safe access to intermediate_output
            intermediate_output = result.get("intermediate_output", {})
            if not "Final Output" in new_msg.content and str(intermediate_output) not in new_msg.content:
                new_msg.content = new_msg.content + "\n\nFinal Output: " + str(intermediate_output)
        else:
            # Fallback if no messages are available
            print(f"⚠️ Warning: No messages found in agent result for {name}")
            new_msg = AIMessage(content=f"Agent {name} completed with no message output", name=name.replace(" ", "_"))
            intermediate_output = result.get("intermediate_output", {})
            if intermediate_output:
                new_msg.content = new_msg.content + "\n\nFinal Output: " + str(intermediate_output)

        if fullshared_memory:
            result["history"] = {
                                    **prev_history,  # Keep the existing history
                                    sup_name: prev_history.get(sup_name, []) + [new_msg],
                                    name: prev_history.get(name, []) + [new_msg],
                                    "all": prev_history.get("all", []) + [new_msg]
                                }
        else:
            result["history"] = {
                                    **prev_history,  # Keep the existing history
                                    sup_name: prev_history.get(sup_name, []) + [new_msg],
                                    name: prev_history.get(name, []) + [new_msg]
                                }
            
        result["messages"] = new_msg
        result["next"] = state["next"]

        # Preserve recent_files state
        if "recent_files" in state:
            result["recent_files"] = state["recent_files"]
        else:
            result["recent_files"] = {}

        # Safely handle background field
        if "background" in result and result["background"] is not None:
            result["background"].name = name
        else:
            # Create a default background if not present
            result["background"] = AIMessage(content=f"Agent {name} task execution", name=name)
            
        result["initial_prompt"] = prev_initial_prompt
        
        # Ensure intermediate_output exists and handle parsing safely
        if "intermediate_output" not in result:
            result["intermediate_output"] = {}
        
        if isinstance(result["intermediate_output"], str):
            try:
                result["intermediate_output"] = ast.literal_eval(result["intermediate_output"])
            except Exception as e:
                try:
                    result["intermediate_output"] = json.loads(result["intermediate_output"])
                except Exception as e2:
                    print(f"⚠️ Could not parse intermediate_output: {str(result['intermediate_output'])}...")
                    print(f"📝 AST error: {e}")
                    print(f"📝 JSON error: {e2}")
                    # Keep the original string if parsing fails, or use empty dict
                    if result["intermediate_output"] == "":
                        result["intermediate_output"] = {}

        # Step 3: Ensure captured intermediate_output is included in final result
        if isinstance(result, dict):
            # If final result has no intermediate_output or is empty, use captured one
            if not result.get("intermediate_output") and captured_intermediate_output:
                print(f"🔧 [Debug] Using captured intermediate_output in final result")
                result["intermediate_output"] = captured_intermediate_output
            # If final result has intermediate_output but captured one is more detailed, merge
            elif captured_intermediate_output and len(str(captured_intermediate_output)) > len(str(result.get("intermediate_output", {}))):
                print(f"🔧 [Debug] Merging captured intermediate_output with final result")
                # Safely handle different types of intermediate_output
                if isinstance(result["intermediate_output"], dict) and isinstance(captured_intermediate_output, dict):
                    result["intermediate_output"].update(captured_intermediate_output)
                else:
                    # If types don't match, directly use captured result
                    result["intermediate_output"] = captured_intermediate_output

        return result





def buildTeam(team_information, react_generator, intermediate_output_desc, int_out_format, index=0, config_override=None):
    team_list = []
    member_info = []
    member_names = []
    color_assign = colors.TerminalColor()
    
    # 🎯 CONFIG OVERRIDE: Apply config override to react_generator if provided
    build_team_id = None  # Track team_id for buildTeam display
    if config_override:
        original_config = react_generator.config
        react_generator.config = config_override
        try:
            round_number = config_override.get('LLM', 'round_number', fallback='unknown')
        except:
            round_number = 'unknown'
        
        #  ADD: Extract team_id from config_override for display
        if isinstance(config_override, dict):
            build_team_id = config_override.get('team_id')
            if 'metadata' in config_override and 'team_id' in config_override['metadata']:
                build_team_id = config_override['metadata']['team_id']
        
        print(f"buildTeam: Applied config override with round_number={round_number}, team_id={build_team_id}")

    print(" " * index * 4, color_assign.colorText(team_information["team"] + " Team/Supervisor:", team_information["team"] + " Supervisor"))

    for key in team_information.keys():
        if isinstance(team_information[key], dict):
            if "team" in team_information[key]:
                team_list.append(buildTeam(team_information[key], react_generator, intermediate_output_desc, int_out_format, index=index+1, config_override=config_override))
                member_names.append(team_information[key]["team"].replace(" ", "_") + " Supervisor")
                member_info.append(team_information[key]["prompt"])
            else:
                team_list.append(react_generator.loadMember(key, team_information[key]["tools"] if "tools" in team_information[key] else [], team_information[key]["prompt"] if "prompt" in team_information[key] else "", team_information["team"].replace(" ", "_") + " Supervisor"))
                member_names.append(key)
                member_info.append(team_information[key]["prompt"] if "prompt" in team_information[key] else "")
                
                # 🎯 ADD: Include team_id in Agent display during buildTeam
                agent_display_name = key + " Agent"
                if build_team_id is not None:
                    agent_display_name = key + f" Team{build_team_id} Agent"
                
                print(" " * (index+1) * 4, color_assign.colorText(agent_display_name, key))
                print(" " * (index+2) * 4, color_assign.colorText("-->" +  str(team_information[key]["tools"]) if "tools" in team_information[key] else "", key))

    agent_team = AgentTeam(None, member_list=team_list, member_info=member_info, member_names=member_names, llm=react_generator.llm, \
                            intermediate_output_desc=intermediate_output_desc, final_output_form=int_out_format, team_name=team_information["team"].replace(" ", "_"), supervisor_name=team_information["return"] if team_information["return"].endswith("Supervisor") else team_information["return"] + " Supervisor")\
                                .createStateGraph(additional_prompt=team_information["additional_prompt"] if "additional_prompt" in team_information else "")
    
    # 🎯 CONFIG RESTORE: Restore original config if override was applied
    if config_override and 'original_config' in locals():
        react_generator.config = original_config
        print(f"🔧 buildTeam: Restored original config")
        
    print(" ")
    return agent_team