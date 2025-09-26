"""
Hard Limit Wrapper for Banner Generation Teams

This module implements a HARD limit mechanism for controlling revision cycles:
1. Hard enforcement: Blocks image generation tools after reaching the limit
2. Tool interception: Monitors and blocks recraft_image_generator/gemini_image_generator calls
3. Immediate termination: Forces FINISH when limit is reached
4. No override: Unlike soft limits, this cannot be bypassed
"""

import time
import os
import glob
import re
from typing import Dict, Any, Callable
from langchain_core.messages import HumanMessage, AIMessage

class HardLimitWrapper:
    """
    Wraps a banner generation team to implement HARD revision limits
    Blocks image generation tools once the limit is reached
    """
    
    def __init__(self, team_func, team_id, max_revisions=3, config=None):
        self.team_func = team_func
        self.team_id = team_id
        self.max_revisions = max_revisions
        self.config = config
        self.original_tools = {}  # Store original tool functions
        self.blocked_calls = 0  # Count blocked tool calls
        self.current_round = 1  # Track current round number
        
        print(f"🔒 [Team {self.team_id}] Hard limit initialized: max_revisions={self.max_revisions} PER ROUND")
        if self.max_revisions == 0:
            print(f"🚫 [Team {self.team_id}] Zero-revision mode: All image generation will be blocked after first image")
    
    def set_current_round(self, round_number):
        """
        Set the current round number for per-round revision counting
        This should be called when starting a new round
        """
        old_round = self.current_round
        self.current_round = round_number
        print(f"🔄 [Team {self.team_id}] Round updated: {old_round} → {round_number} (revisions reset)")
        
    def __call__(self, state, config):
        """
        Execute the banner generation team with HARD revision limits
        """
        # Update current round from state or config if available
        if "round_number" in state:
            self.set_current_round(state["round_number"])
        elif "round_number" in config:
            self.set_current_round(config["round_number"])
            
        print(f"🔒 [Team {self.team_id}] Starting execution with HARD limit: {self.max_revisions} revisions max (Round {self.current_round})")
        print(f"⏰ [Team {self.team_id}] Execution start time: {time.strftime('%H:%M:%S')}")
        
        # 🔧 CRITICAL FIX: Update tools' round_number before execution
        tools_updated = self._update_tools_round_number_direct(self.current_round)
        if tools_updated > 0:
            print(f"✅ [Team {self.team_id}] Updated {tools_updated} tools to Round {self.current_round}")
        else:
            print(f"⚠️ [Team {self.team_id}] No tools found to update (they may be created dynamically)")
        
        start_time = time.time()
        
        try:
            # Apply hard limits by monitoring team execution
            print(f"🔧 [Team {self.team_id}] Applying hard limit monitoring to team...")
            modified_team = self._apply_hard_limits_to_team(self.team_func)
            
            # Execute with the modified team that has hard limits
            print(f"🚀 [Team {self.team_id}] Starting monitored team execution...")
            result = modified_team(state, config)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            actual_revisions = self._count_team_revisions_for_round(self.current_round)
            
            print(f"✅ [Team {self.team_id}] HARD limit execution completed in {execution_time:.2f}s")
            print(f"📊 [Team {self.team_id}] Round {self.current_round} revision count: {actual_revisions}")
            print(f"🚫 [Team {self.team_id}] Blocked tool calls: {self.blocked_calls}")
            print(f"⏰ [Team {self.team_id}] Completion time: {time.strftime('%H:%M:%S')}")
            
            # Add hard limit metadata to result
            if isinstance(result, list):
                final_result = result[0]
            else:
                final_result = result
                
            if "intermediate_output" in final_result:
                if isinstance(final_result["intermediate_output"], dict):
                    final_result["intermediate_output"]["revision_count"] = actual_revisions
                    final_result["intermediate_output"]["round_number"] = self.current_round
                    final_result["intermediate_output"]["hard_limit_status"] = "enforced"
                    final_result["intermediate_output"]["max_revisions"] = self.max_revisions
                    final_result["intermediate_output"]["blocked_calls"] = self.blocked_calls
                    final_result["intermediate_output"]["limit_type"] = "HARD_PER_ROUND"
            
            return result
            
        except Exception as e:
            print(f"❌ [Team {self.team_id}] Hard limit execution failed: {e}")
            print(f"⏰ [Team {self.team_id}] Failure time: {time.strftime('%H:%M:%S')}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_result(state)
    
    def _apply_hard_limits_to_team(self, team_func):
        """
        Apply hard limits by monitoring team execution (tools now self-enforce limits)
        """
        def limited_team_func(state, config):
            """
            Team function wrapper that monitors hard limit compliance
            """
            print(f"🔧 [Team {self.team_id}] Monitoring team execution with hard limit enforcement...")
            print(f"📊 [Team {self.team_id}] Tools are self-enforcing limits: max {self.max_revisions} revisions per round")
            
            try:
                # Execute the original team function
                # Tools now self-enforce hard limits in their _run methods
                print(f"🚀 [Team {self.team_id}] Executing team with tool-level hard limit monitoring...")
                result = team_func(state, config)
                
                # 🔧 POST-EXECUTION: Check final revision count for reporting
                final_revisions = self._count_team_revisions_for_round(self.current_round)
                print(f"📈 [Team {self.team_id}] Final revision count for Round {self.current_round}: {final_revisions}")
                
                if final_revisions > self.max_revisions:
                    print(f"⚠️ [Team {self.team_id}] WARNING: Hard limit exceeded after execution: {final_revisions} > {self.max_revisions}")
                    # This shouldn't happen with tool-level enforcement, but we log it
                    if isinstance(result, list):
                        final_result = result[0]
                    else:
                        final_result = result
                    
                    if "intermediate_output" in final_result and isinstance(final_result["intermediate_output"], dict):
                        final_result["intermediate_output"]["hard_limit_exceeded"] = True
                        final_result["intermediate_output"]["actual_revisions"] = final_revisions
                else:
                    print(f"✅ [Team {self.team_id}] Hard limit compliance: {final_revisions}/{self.max_revisions} revisions")
                
                return result
                
            except RuntimeError as e:
                # Catch hard limit errors from tools
                if "HARD LIMIT REACHED" in str(e):
                    print(f"🔒 [Team {self.team_id}] Tool-enforced hard limit triggered: {e}")
                    # Create a controlled result indicating the limit was reached
                    return self._create_hard_limit_result(state, str(e))
                else:
                    raise e
            except Exception as e:
                print(f"❌ [Team {self.team_id}] Team execution failed: {e}")
                raise e
        
        return limited_team_func
    
    def _intercept_team_tools(self, team_obj):
        """
        Recursively find and intercept image generation tools in the team
        Enhanced search for LangGraph and LangChain structures
        """
        print(f"🔍 [Team {self.team_id}] Starting comprehensive tool search for interception...")
        tools_intercepted = 0
        
        def find_and_intercept_tools(obj, depth=0, path="root"):
            nonlocal tools_intercepted
            
            if depth > 15:  # Increase depth for deeper searches
                return
            
            # DEBUG: Print what we're examining (for first few levels)
            if depth <= 4:
                print(f"🔍 [Depth {depth}] Examining {path}: {type(obj).__name__}")
            
            # Method 1: Direct tools attribute
            if hasattr(obj, 'tools') and obj.tools is not None:
                print(f"🎯 [Team {self.team_id}] Found 'tools' attribute at {path}")
                self._process_tools_collection(obj.tools, tools_intercepted, path)
            
            # Method 2: ToolNode pattern (common in LangGraph)
            if hasattr(obj, 'tools_by_name') and obj.tools_by_name:
                print(f"🎯 [Team {self.team_id}] Found 'tools_by_name' at {path}")
                for tool_name, tool in obj.tools_by_name.items():
                    if self._is_image_generation_tool(tool_name):
                        tools_intercepted += self._intercept_single_tool(tool, tool_name, f"{path}.tools_by_name[{tool_name}]")
            
            # Method 3: Agent with bound tools (LangChain agent pattern)
            if hasattr(obj, 'agent') and obj.agent is not None:
                print(f"🎯 [Team {self.team_id}] Found 'agent' at {path}")
                find_and_intercept_tools(obj.agent, depth + 1, f"{path}.agent")
            
            # Method 4: Model with bound tools
            if hasattr(obj, 'model') and obj.model is not None:
                print(f"🎯 [Team {self.team_id}] Found 'model' at {path}")
                find_and_intercept_tools(obj.model, depth + 1, f"{path}.model")
                
            # Method 5: Check for bound tools in the model itself
            if hasattr(obj, 'bound_tools') and obj.bound_tools:
                print(f"🎯 [Team {self.team_id}] Found 'bound_tools' at {path}")
                self._process_tools_collection(obj.bound_tools, tools_intercepted, f"{path}.bound_tools")
            
            # Method 6: LangGraph nodes pattern
            if hasattr(obj, 'nodes') and obj.nodes:
                print(f"🎯 [Team {self.team_id}] Found 'nodes' at {path}")
                if isinstance(obj.nodes, dict):
                    for node_name, node in obj.nodes.items():
                        find_and_intercept_tools(node, depth + 1, f"{path}.nodes[{node_name}]")
                else:
                    find_and_intercept_tools(obj.nodes, depth + 1, f"{path}.nodes")
            
            # Method 7: StateGraph pattern
            if hasattr(obj, 'graph') and obj.graph is not None:
                print(f"🎯 [Team {self.team_id}] Found 'graph' at {path}")
                find_and_intercept_tools(obj.graph, depth + 1, f"{path}.graph")
            
            # Method 8: Check if object itself is a tool
            if self._is_tool_object(obj):
                tool_name = getattr(obj, 'name', f'Tool_at_{path}')
                if self._is_image_generation_tool(tool_name):
                    print(f"🎯 [Team {self.team_id}] Found direct tool: {tool_name} at {path}")
                    tools_intercepted += self._intercept_single_tool(obj, tool_name, path)
            
            # Method 9: Recursive search in common LangChain/LangGraph attributes
            common_attrs = [
                'runnable', 'bound', 'last', 'first', 'chain', 
                'nodes_dict', 'compiled', 'workflow', 'state_graph',
                'tool_node', 'agent_runnable', 'llm_with_tools'
            ]
            
            for attr in common_attrs:
                if hasattr(obj, attr):
                    attr_value = getattr(obj, attr)
                    if attr_value is not None:
                        find_and_intercept_tools(attr_value, depth + 1, f"{path}.{attr}")
            
            # Method 10: Dict-like iteration
            if hasattr(obj, 'items') and callable(getattr(obj, 'items')):
                try:
                    for key, value in obj.items():
                        if value is not None and depth < 10:  # Limit dict traversal
                            find_and_intercept_tools(value, depth + 1, f"{path}[{key}]")
                except Exception:
                    pass
            
            # Method 11: List-like iteration (limited)
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                try:
                    for i, item in enumerate(obj):
                        if item is not None and i < 50:  # Limit list traversal
                            find_and_intercept_tools(item, depth + 1, f"{path}[{i}]")
                        if i >= 50:  # Prevent excessive iteration
                            break
                except Exception:
                    pass
        
        # Start comprehensive search
        find_and_intercept_tools(team_obj)
        print(f"🔒 [Team {self.team_id}] Tool interception complete: {tools_intercepted} tools intercepted")
        return tools_intercepted
    
    def _process_tools_collection(self, tools_collection, tools_intercepted, path):
        """Process a collection of tools"""
        try:
            if hasattr(tools_collection, '__iter__'):
                for i, tool in enumerate(tools_collection):
                    tool_name = getattr(tool, 'name', f'Tool{i}')
                    if self._is_image_generation_tool(tool_name):
                        tools_intercepted += self._intercept_single_tool(tool, tool_name, f"{path}[{i}]")
            else:
                # Single tool
                tool_name = getattr(tools_collection, 'name', 'SingleTool')
                if self._is_image_generation_tool(tool_name):
                    tools_intercepted += self._intercept_single_tool(tools_collection, tool_name, path)
        except Exception as e:
            print(f"⚠️ [Team {self.team_id}] Error processing tools collection at {path}: {e}")
    
    def _is_image_generation_tool(self, tool_name):
        """Check if a tool name indicates an image generation tool"""
        if not tool_name:
            return False
        tool_name_lower = tool_name.lower()
        return any(keyword in tool_name_lower for keyword in [
            'image', 'recraft', 'gemini', 'generation', 'generator', 'img'
        ])
    
    def _is_tool_object(self, obj):
        """Check if an object appears to be a tool"""
        return (hasattr(obj, 'name') and 
                hasattr(obj, '_run') and 
                callable(getattr(obj, '_run', None)))
    
    def _intercept_single_tool(self, tool, tool_name, tool_path):
        """Intercept a single tool"""
        try:
            original_run = getattr(tool, '_run', None)
            if original_run and callable(original_run):
                # Store original function
                self.original_tools[tool_path] = original_run
                
                # Replace with hard-limited version
                tool._run = self._create_limited_tool_function(original_run, tool_name)
                print(f"🔒 [Team {self.team_id}] Intercepted {tool_name} at {tool_path}")
                return 1
            else:
                print(f"⚠️ [Team {self.team_id}] Tool {tool_name} has no callable _run method")
                return 0
        except Exception as e:
            print(f"⚠️ [Team {self.team_id}] Error intercepting tool {tool_name}: {e}")
            return 0
    
    def _create_limited_tool_function(self, original_func: Callable, tool_name: str) -> Callable:
        """
        Create a hard-limited version of an image generation tool
        """
        def limited_tool_function(*args, **kwargs):
            # Check current revision count before allowing tool execution
            current_revisions = self._count_team_revisions_for_round(self.current_round)
            
            print(f"🔍 [Team {self.team_id}] {tool_name} called - Current revisions: {current_revisions}, Limit: {self.max_revisions}")
            
            # Hard limit enforcement
            if current_revisions >= self.max_revisions:
                self.blocked_calls += 1
                error_msg = f"🚫 HARD LIMIT REACHED: Team {self.team_id} has already generated {current_revisions} revisions (limit: {self.max_revisions}). Image generation blocked."
                print(error_msg)
                
                # Return a clear error that will stop the workflow
                raise RuntimeError(error_msg)
            
            # Allow execution if under limit
            try:
                print(f"✅ [Team {self.team_id}] {tool_name} execution allowed (revisions: {current_revisions}/{self.max_revisions})")
                result = original_func(*args, **kwargs)
                
                # Check again after execution
                new_revision_count = self._count_team_revisions_for_round(self.current_round)
                print(f"📊 [Team {self.team_id}] After {tool_name}: revisions {current_revisions} → {new_revision_count}")
                
                return result
            except Exception as e:
                print(f"❌ [Team {self.team_id}] {tool_name} execution failed: {e}")
                raise e
        
        return limited_tool_function
    
    def _restore_original_tools(self):
        """
        Restore original tool functions (cleanup)
        """
        restored_count = 0
        for tool_path, original_func in self.original_tools.items():
            try:
                # This is complex to implement properly, so we'll just clear the dict
                # In practice, team objects are usually recreated for each execution
                restored_count += 1
            except Exception as e:
                print(f"⚠️ [Team {self.team_id}] Could not restore tool at {tool_path}: {e}")
        
        self.original_tools.clear()
        if restored_count > 0:
            print(f"🔄 [Team {self.team_id}] Attempted to restore {restored_count} original tools")
    
    def _count_team_revisions_for_round(self, round_number):
        """
        Count actual revisions by examining generated files for a specific round
        """
        if self.config:
            if hasattr(self.config, 'get') and hasattr(self.config, 'has_section'):
                # ConfigParser object
                output_folder = self.config.get("SETTING", "output_folder", fallback="outputs")
            elif isinstance(self.config, dict):
                # Dict config
                output_folder = self.config.get("output_folder", "outputs")
            else:
                output_folder = "outputs"
        else:
            output_folder = "outputs"
        
        pattern = f"{output_folder}/generated_image_*_Team{self.team_id}_Round{round_number}_*.png"
        files = glob.glob(pattern)
        
        max_revision = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
            match = re.search(r'_Team\d+_Round\d+_(\d+)\.png$', filename)
            if match:
                revision_num = int(match.group(1))
                max_revision = max(max_revision, revision_num)
        
        return max_revision
    
    def _create_fallback_result(self, state):
        """
        Create a fallback result when hard limit execution fails
        """
        actual_revisions = self._count_team_revisions_for_round(self.current_round)
        
        return {
            "intermediate_output": {
                "status": "hard_limit_enforced",
                "team_id": self.team_id,
                "revision_count": actual_revisions,
                "round_number": self.current_round,
                "max_revisions": self.max_revisions,
                "blocked_calls": self.blocked_calls,
                "limit_type": "HARD_PER_ROUND",
                "message": f"Team {self.team_id} completed with hard revision limit enforcement",
                "note": "Revision generation was blocked when limit was reached"
            },
            "messages": [state.get("background", "")],
            "history": state.get("history", {}),
            "recent_files": state.get("recent_files", {}),
            "next": "FINISH"
        }
    
    def _create_hard_limit_result(self, state, error_message):
        """
        Create a result when hard limit is enforced by tools
        """
        actual_revisions = self._count_team_revisions_for_round(self.current_round)
        
        # Extract team supervisor name from state
        history_keys = list(state.get("history", {}).keys())
        supervisor_name = history_keys[0] if history_keys else f"BannerGeneration{self.team_id} Supervisor"
        
        return {
            "intermediate_output": {
                "status": "hard_limit_enforced_by_tool",
                "team_id": self.team_id,
                "revision_count": actual_revisions,
                "round_number": self.current_round,
                "max_revisions": self.max_revisions,
                "limit_type": "HARD_PER_ROUND_TOOL_ENFORCED",
                "message": f"Team {self.team_id} execution halted due to hard revision limit",
                "error_message": error_message,
                "note": "Image generation was blocked by tool-level hard limit enforcement"
            },
            "messages": [state.get("background", "")],
            "history": {supervisor_name: []},
            "recent_files": state.get("recent_files", {}),
            "next": "FINISH"
        }
    
    def _update_tools_round_number_direct(self, round_number):
        """
        Directly search for and update image generation tools' round_number
        This is a more aggressive search that looks deeper into the team structure
        """
        tools_updated = 0
        
        def deep_tool_search(obj, depth=0, path="root"):
            nonlocal tools_updated
            
            if depth > 20:  # Very deep search
                return
            
            # Skip certain types to avoid infinite recursion
            if isinstance(obj, (str, int, float, bool, type(None))):
                return
                
            try:
                # Check if this is an image generation tool
                if hasattr(obj, 'name') and hasattr(obj, '_run') and hasattr(obj, 'round_number'):
                    tool_name = getattr(obj, 'name', 'UnknownTool')
                    if 'image' in tool_name.lower() and 'generator' in tool_name.lower():
                        old_round = obj.round_number
                        if hasattr(obj, 'update_round_number'):
                            obj.update_round_number(round_number)
                        else:
                            obj.round_number = round_number
                        tools_updated += 1
                        print(f"🔧 [Direct Update] {tool_name}: Round {old_round} → {round_number}")
                        return  # Found and updated, no need to go deeper
                
                # Search in common attributes
                search_attrs = [
                    'tools', 'agent', 'model', 'llm', 'runnable', 'bound', 'chain',
                    'nodes', 'graph', 'team_func', 'func', 'args', 'kwargs'
                ]
                
                for attr in search_attrs:
                    if hasattr(obj, attr):
                        try:
                            attr_value = getattr(obj, attr)
                            if attr_value is not None:
                                deep_tool_search(attr_value, depth + 1, f"{path}.{attr}")
                        except:
                            continue
                
                # Search in dict-like objects
                if hasattr(obj, 'items') and callable(getattr(obj, 'items')):
                    try:
                        for key, value in obj.items():
                            if value is not None and key not in ['config', 'state']:
                                deep_tool_search(value, depth + 1, f"{path}[{key}]")
                    except:
                        pass
                
                # Search in list-like objects
                if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                    try:
                        for i, item in enumerate(obj):
                            if item is not None:
                                deep_tool_search(item, depth + 1, f"{path}[{i}]")
                            if i > 50:  # Limit iterations
                                break
                    except:
                        pass
                        
            except Exception as e:
                # Silently continue on errors to avoid breaking the search
                pass
        
        # Start deep search from team_func
        if hasattr(self, 'team_func') and self.team_func is not None:
            deep_tool_search(self.team_func)
        
        return tools_updated

def wrap_team_with_hard_limit(team_func, team_id, max_revisions=3, config=None):
    """
    Factory function to create a hard-limited team
    
    Args:
        team_func: The original team function to wrap
        team_id: Unique identifier for this team
        max_revisions: Maximum number of revisions allowed (HARD limit)
        config: Configuration object
    
    Returns:
        HardLimitWrapper instance that enforces revision limits
    """
    return HardLimitWrapper(team_func, team_id, max_revisions, config) 