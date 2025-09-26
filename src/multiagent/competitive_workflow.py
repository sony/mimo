import sys
import json
import os
import random
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
sys.path.append(os.path.dirname(__file__))
import agent_team as agent_team
from llm import LLM
from hard_limit_wrapper import wrap_team_with_hard_limit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import prompts.competitive_multimodal as CompetitivePrompt
import threading
import concurrent.futures
import time
import configparser
import pdb
class CompetitiveWorkflow:
    """
    Manages the competitive banner generation workflow with configurable number of parallel teams
    and elimination rounds judged by a Judge Agent. Uses HARD limits for revision control.
    """
    
    def __init__(self, config, llm=None, logo_path=None, use_dynamic_styles=False):
        self.config = config
        self.llm = llm if llm else LLM(config).llm
        self.logo_path = logo_path
        self.use_dynamic_styles = use_dynamic_styles
        self.judge_combination = self.config.get("SETTING", "judge_combination", fallback="all")
        # 🎨 TEAM CONFIGURATION (read from config file)
        # Control the number of competing banner teams (1-10, support more teams)
        try:
            config_num_teams = int(self.config.get("LLM", "num_teams", fallback=5))
        except:
            config_num_teams = 5  # Default fallback
        self.num_teams = max(1, min(10, config_num_teams))  # Clamp between 1 and 10
        
        # 🔒 HARD REVISION CONTROL PARAMETER
        # This HARD limit blocks image generation tools after reaching the limit
        # Adjust this value to control revision iterations per team (default: 3)
        # Can be controlled via config file: [LLM] max_revisions_per_team = 2
        try:
            self.max_revisions_per_team = int(self.config.get("LLM", "max_revisions_per_team", fallback=3))
        except:
            self.max_revisions_per_team = 3  # Default fallback
        
        # 🔧 IMAGE EDIT CONTROL PARAMETER
        # This controls how many times GeminiTool can edit images per round
        # Set to 0 to disable editing completely for faster workflow testing
        try:
            self.max_image_edits = int(self.config.get("LLM", "max_image_edits", fallback=0))
        except:
            self.max_image_edits = 0  # Default to no edits
        
        # Team IDs will be determined by dynamic style selection
        # For dynamic styles: determined by which styles are selected from candidates
        # For static styles: use consecutive IDs [1, 2, 3, ...]
        self.selected_team_ids = None  # Will be set later
        self.active_banners = None  # Will be set after team IDs are determined
        self.round_number = 1
        self.elimination_history = []
        self.banner_results = {}  # Store results for each banner
        self.judge_evaluations = []  # Store all judge evaluations
        self._results_lock = threading.Lock()  # Thread safety for results
        
        # 🧠 PERSISTENT TEAM INSTANCES AND STATES
        # Maintain active banner teams with their memory and states
        self.active_teams = {}  # team_id -> team instance 
        self.team_states = {}   # team_id -> current team state
        self.team_configs = {}  # team_id -> team configuration
        self.prompt_template = None  # Will be set in first round
        
        # Design philosophy mapping (will be updated if using dynamic styles)
        self.design_philosophies = {
            1: "Bold & Eye-catching",
            2: "Elegant & Professional", 
            3: "Modern & Minimalist",
            4: "Vibrant & Energetic",
            5: "Classic & Trustworthy",
            6: "Creative & Artistic",
            7: "Tech & Futuristic", 
            8: "Warm & Friendly",
            9: "Luxury & Premium",
            10: "Fresh & Natural"
        }
        
        # Store dynamic styles if generated
        self.dynamic_styles = None
        
        print(f"🔧 CompetitiveWorkflow initialized with {self.num_teams} teams (config: num_teams={config_num_teams})")
        print(f"🎯 Team IDs will be determined by style selection")
        print(f"🔒 HARD limit revisions per team: {self.max_revisions_per_team}")
        print(f"✂️ EDIT limit per team: {self.max_image_edits} (0 = no editing)")
        print(f"🧠 Persistent team memory: ENABLED")
        print("📝 To change team count, modify num_teams in config/config_llm.ini")
        
    def execute_competitive_generation(self, item_description):
        """
        Execute the full competitive banner generation workflow
        """
        print("🚀 Starting Competitive Banner Generation Workflow")
        print(f"📝 Item: {item_description}")
        print(f"🎯 Teams competing: {self.num_teams}")
        print("=" * 60)
        
        try:
            # Phase 1: Initial parallel generation (CONCURRENT)
            print("\n🎨 PHASE 1: Initial Banner Generation")
            self._generate_initial_banners_concurrent(item_description)
            
            # Check if we have any valid banners to proceed
            valid_banners = [team_id for team_id in self.selected_team_ids 
                           if team_id in self.banner_results and "error" not in self.banner_results[team_id]]
            
            print(f"\n📊 Phase 1 Complete - Valid banners: {valid_banners}")
            
            if len(valid_banners) == 1:
                print("⚠️ Not enough valid banners for competition. Need at least 2 banners.")
                return self._finalize_results()
            
            # Update active banners to only include valid ones
            self.active_banners = valid_banners  # Keep as list for index access
            print(f"🎯 Active banners for competition: {self.active_banners}")
            
            # Phase 2: Elimination rounds (eliminate until 1 remains)
            print(f"\n⚔️ PHASE 2: Elimination Rounds")
            while len(self.active_banners) > 1:
                print(f"\n⚔️ Round {self.round_number + 1}: Evaluation & Elimination")
                print(f"🎯 Active Banners: {self.active_banners}")
                
                # Judge evaluation and elimination
                print("👨‍⚖️ Starting judge evaluation...")
                judge_result = self._execute_judge_evaluation()
                
                # Process elimination
                if len(self.active_banners) > 1:
                    print("🔄 Processing elimination...")
                    self._process_elimination(judge_result)
                
                # Improve remaining banners based on feedback (this will increment round_number)
                if len(self.active_banners) > 1:
                    print("🔧 Improving remaining banners...")
                    self._improve_remaining_banners(judge_result)
                elif len(self.active_banners) == 1:
                    print("🏆 Only one banner remaining - competition completed!")
                    break
                
                print(f"✅ Round {self.round_number} complete. Remaining: {self.active_banners}")
                print(f"🎯 Current round_number: {self.round_number}")
                
                # Safety check to prevent infinite loops
                if self.round_number > 10:
                    print("⚠️ Maximum rounds reached, breaking...")
                    break
            
            # Phase 3: Declare winner and final results
            print(f"\n🏆 PHASE 3: Final Results")
            return self._finalize_results()
            
        except Exception as e:
            print(f" Error in competitive workflow: {e}")
            import traceback
            traceback.print_exc()
            return self._finalize_results()
    
    def _generate_single_banner(self, team_id, item_description, prompt_template):
        """
        Generate a single banner for a specific team (thread-safe)
        Creates and stores persistent team instances for later reuse
        """
        try:
            print(f"\n🔨 [Thread {team_id}] Starting Banner #{team_id} generation...")
            print(f"🕒 [Thread {team_id}] Start time: {time.strftime('%H:%M:%S')}")
            
            # Create individual team prompt
            team_key = f"BannerTeam{team_id}"
            team_config = prompt_template["team"][team_key]
            print(f"📝 [Thread {team_id}] Team config loaded: {team_key}")
            
            # Store team configuration for later reuse
            self.team_configs[team_id] = team_config
            
            # Build the team with team-specific configuration
            print(f"🔧 [Thread {team_id}] Building ReactAgent...")
            react_generator = agent_team.ReactAgent(
                intermediate_output_desc=prompt_template["intermediate_output_desc"], 
                config=self.config, 
                llm=self.llm
            )
            
            # 🎯 CRITICAL: Inject team_id into react_generator for unique naming
            react_generator.team_id = team_id
            print(f"🎯 [Thread {team_id}] Team ID injected: {team_id}")
            
            # 🎯 ROUND NUMBER INJECTION: Add round number to config for tools
            # We'll pass the round_number directly to tools instead of modifying config
            print(f"🔧 [Thread {team_id}] Will inject round_number={self.round_number} into tools")
            
            # Create a modified team config with team-specific agent names
            print(f"🔧 [Thread {team_id}] Modifying agent names for team-specific naming...")
            modified_team_config = self._modify_agent_names_for_team(team_config, team_id)
            
            # Build the base team
            print(f"🏗️ [Thread {team_id}] Building base team structure...")
            base_team = agent_team.buildTeam(
                modified_team_config, 
                react_generator, 
                prompt_template["intermediate_output_desc"], 
                prompt_template["int_out_format"]
            )
            
            # Wrap with hard limit functionality
            print(f"🔒 [Thread {team_id}] Applying hard limit wrapper (max: {self.max_revisions_per_team} revisions)...")
            team = wrap_team_with_hard_limit(base_team, team_id, max_revisions=self.max_revisions_per_team, config=self.config)
            
            # 🎯 DIRECT TOOL UPDATE: Update round_number on all tools after team creation
            print(f"🔧 [Thread {team_id}] Updating tools to Round {self.round_number}...")
            self._update_team_tools_round_number(team, self.round_number)
            
            # 🧠 STORE TEAM INSTANCE FOR PERSISTENT MEMORY
            self.active_teams[team_id] = team
            print(f"🧠 [Thread {team_id}] Team instance stored for future rounds")
            
            # Execute team with hard revision limit
            team_prompt = team_config["prompt"].replace("{item}", item_description)
            
            # Create background message for the banner team
            background_content = f"Generate AD Banner for: {item_description}. Design Philosophy: {team_config['additional_prompt']}. This is part of a competitive banner generation workflow where {self.num_teams} teams create different design approaches. Your team ID is {team_id}. HARD LIMIT ENFORCED: Image generation tools will be BLOCKED after {self.max_revisions_per_team} revisions."
            
            # 🧠 INITIAL TEAM STATE
            initial_state = {
                "history": {
                    f"BannerGeneration{team_id} Supervisor": [HumanMessage(content=team_prompt)], 
                    "all": [HumanMessage(content=team_prompt)]
                }, 
                "intermediate_output": {},
                "recent_files": {},  # Add file tracking support
                "background": AIMessage(content=background_content),
                "team_id": team_id,  # 🎯 ADD: Pass team ID in state
                "round_number": self.round_number  # 🎯 ADD: Pass round number in state
            }
            
            print(f"🚀 [Thread {team_id}] Starting team execution with max {int(self.config['LLM']['max_attempts'])} recursion limit...")
            start_time = time.time()
            
            # Execute with hard limit wrapper
            result = team(
                state=initial_state,
                config={
                    "recursion_limit": int(self.config["LLM"]["max_attempts"]),
                    "team_id": team_id,  # For debugging
                    "round_number": self.round_number,  # 🎯 ADD: Pass round_number to config
                    "metadata": {  # 🎯 ADD: Include round_number in metadata for tools
                        "round_number": self.round_number,
                        "team_id": team_id,
                        "workflow_phase": "initial_generation"
                    }
                }
            )
            
            print(f"[Thread {team_id}] Team execution completed successfully")
            
            # 🧠 STORE TEAM STATE FOR NEXT ROUND
            if isinstance(result, list):
                final_state = result[0]
                intermediate_output = final_state["intermediate_output"]
            else:
                final_state = result
                intermediate_output = result["intermediate_output"]
            
            # 🔧 FIX: Ensure saved state has all required fields for next round
            # The team execution result might not have all the initial state fields
            preserved_state = {
                "history": final_state.get("history", {}),
                "messages": final_state.get("messages", []),
                "intermediate_output": final_state.get("intermediate_output", {}),
                "recent_files": final_state.get("recent_files", {}),
                "next": final_state.get("next", "FINISH"),
                "team_id": team_id,
                "round_number": self.round_number
            }
            
            # 🚨 CRITICAL: Ensure background field is properly preserved as AIMessage
            if "background" in final_state and final_state["background"] is not None:
                # If background exists and is AIMessage, preserve it
                if hasattr(final_state["background"], "content"):
                    preserved_state["background"] = final_state["background"]
                else:
                    # If background exists but wrong type, convert it
                    background_content = str(final_state["background"])
                    preserved_state["background"] = AIMessage(content=background_content)
                    print(f"🔧 [Team {team_id}] Converted background to AIMessage type")
            else:
                # If no background in result, create a proper one for next round
                background_content = f"Banner #{team_id} generation completed in Round {self.round_number}. Design Philosophy: {self.design_philosophies.get(team_id, 'Unknown')}. Team ID: {team_id}."
                preserved_state["background"] = AIMessage(content=background_content)
                print(f"🔧 [Team {team_id}] Created missing background field for state preservation")
            
            # Store both the result and the preserved state
            with self._results_lock:
                self.banner_results[team_id] = intermediate_output
                self.team_states[team_id] = preserved_state  # 🎯 Save complete state structure
                    
            print(f"   [Team {team_id}] State saved with all required fields:")
            print(f"   - background: {type(preserved_state['background'])}")
            print(f"   - history: {type(preserved_state['history'])} ({len(preserved_state['history'])} keys)")
            print(f"   - messages: {type(preserved_state['messages'])}")
            print(f"   - team_id: {preserved_state['team_id']}")
            print(f"   - round_number: {preserved_state['round_number']}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"   [Thread {team_id}] Banner #{team_id} completed in {execution_time:.2f}s")
            print(f"   [Thread {team_id}] Team instance and state saved for future rounds")
            
            return team_id, "success"
            
        except Exception as e:
            print(f"   [Thread {team_id}] Error generating Banner #{team_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Thread-safe error storage
            with self._results_lock:
                self.banner_results[team_id] = {"error": str(e)}
            return team_id, f"error: {e}"

    def _update_team_tools_round_number(self, team, round_number):
        """
        🎯 SIMPLIFIED: Tools now get round_number from config automatically
        This function is kept for compatibility but tools no longer need manual updates
        """
        print(f"   [SIMPLIFIED] Tools will get round_number={round_number} from config automatically")
        
        # Update HardLimitWrapper's current round (still needed for revision counting)
        if hasattr(team, 'set_current_round'):
            team.set_current_round(round_number)
            print(f"   HardLimitWrapper updated to round {round_number}")
            return 1  # Return 1 to indicate update completed
        
        return 0

    def _modify_agent_names_for_team(self, team_config, team_id):
        """
        Modify agent names to include team ID for unique file naming
        This ensures that each Banner team's agents have unique identifiers
        """
        import copy
        modified_config = copy.deepcopy(team_config)
        
        # 🎯 ENHANCED: Modify agent names to include team information
        # This is critical for file naming: generated_image_ImageResearcher_Team{ID}_Round{R}_{Rev}.png
        def modify_nested_config(config_dict, team_id):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # If this is an agent definition (has prompt and tools)
                    if "prompt" in value and "tools" in value:
                        # For ImageResearcher, ensure the name includes Team info
                        if key == "ImageResearcher":
                            config_dict[key] = value  # Keep the original config, team ID will be passed via agent name
                    elif "team" in value:
                        # This is a sub-team, recursively modify
                        modify_nested_config(value, team_id)
        
        modify_nested_config(modified_config, team_id)
        return modified_config

    def _generate_initial_banners_concurrent(self, item_description):
        """
        Generate initial banners using configurable number of parallel teams (CONCURRENT EXECUTION)
        Creates persistent team instances that will be reused in subsequent rounds
        """
        print(f"🎨 Round 1: Initial Banner Generation ({self.num_teams} CONCURRENT teams)")
        
        # Get competitive prompt template with dynamic style support
        # This will determine the actual team IDs based on style selection
        self.prompt_template = CompetitivePrompt.getCompetitivePrompts(
            "AD Banner Generation", 
            item_description, 
            self.llm, 
            selected_team_ids=None,  # Let it determine team IDs based on styles
            num_teams=self.num_teams,
            config=self.config,
            logo_path=self.logo_path,
            use_dynamic_styles=self.use_dynamic_styles
        )
        # Extract the actual team IDs from the generated prompt template
        self.selected_team_ids = []
        for team_key in self.prompt_template["team"].keys():
            if team_key.startswith("BannerTeam"):
                team_id = int(team_key.replace("BannerTeam", ""))
                self.selected_team_ids.append(team_id)
        self.selected_team_ids = sorted(self.selected_team_ids)
        self.active_banners = self.selected_team_ids.copy()
        
        print(f"   Starting {self.num_teams} teams simultaneously...")
        print(f"   Selected team IDs: {self.selected_team_ids}")
        
        # Update design philosophies if dynamic styles were generated
        if self.use_dynamic_styles and "BannerTeam1" in self.prompt_template["team"]:
            print("   Updating design philosophies with dynamic styles...")
            for i, team_id in enumerate(self.selected_team_ids, 1):
                team_key = f"BannerTeam{team_id}"
                if team_key in self.prompt_template["team"] and "style_info" in self.prompt_template["team"][team_key]:
                    style_info = self.prompt_template["team"][team_key]["style_info"]
                    self.design_philosophies[team_id] = f"{style_info['style_name']}: {style_info['philosophy']}"
                    print(f"   Team {team_id}: {style_info['style_name']}")
            
            print("   Design philosophies updated with AI-generated styles")
        
        # Use ThreadPoolExecutor for concurrent execution
        start_time = time.time()
        completed_teams = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_teams) as executor:
                # Submit banner generation tasks for selected teams only
                future_to_team = {
                    executor.submit(self._generate_single_banner, team_id, item_description, self.prompt_template): team_id 
                    for team_id in self.selected_team_ids
                }
                
                # 🔧 ENHANCED MONITORING: Track execution status and show progress
                print(f"📊 Monitoring {len(future_to_team)} concurrent team executions...")
                running_teams = set(self.selected_team_ids)
                completed_teams = []
                
                # 🔧 NEW: Start a monitoring thread for periodic status updates
                import threading
                monitoring_active = threading.Event()
                monitoring_active.set()
                termination_requested = threading.Event()  # 🔧 ADD: Signal for early termination
                
                # Track when single team started running alone
                single_team_start_time = [None]  # Use list to make it mutable in closure
                last_running_count = [len(running_teams)]  # Use list to make it mutable in closure
                
                def periodic_status_monitor():
                    """Background thread to report execution status every 30 seconds and terminate last stuck team"""
                    import time
                    last_report_time = start_time
                    
                    while monitoring_active.is_set():
                        time.sleep(30)  # Report every 30 seconds
                        if not monitoring_active.is_set():
                            break
                            
                        current_time = time.time()
                        elapsed = current_time - start_time
                        current_running_count = len(running_teams)
                        
                        print(f"\n   [STATUS REPORT] Elapsed: {elapsed:.1f}s")
                        print(f"    Completed: {len(completed_teams)}/{self.num_teams} teams")
                        
                        if running_teams:
                            print(f"   ⏳ Still running: Teams {sorted(list(running_teams))}")
                            print(f"   ⏱️ Each running for: {elapsed:.1f}s")
                            
                            # 🎯 DETECT SINGLE TEAM: Check if only one team is left running
                            if current_running_count == 1 and last_running_count[0] > 1:
                                # Just transitioned to single team running
                                single_team_start_time[0] = current_time
                                remaining_team = list(running_teams)[0]
                                print(f"   🎯 SINGLE TEAM DETECTED: Only Team {remaining_team} is still running")
                                print(f"      Starting 60-second timeout monitoring for stuck team...")
                            
                            #  TERMINATE LAST STUCK TEAM: If only one team running for > 60 seconds
                            if current_running_count == 1 and single_team_start_time[0] is not None:
                                single_team_elapsed = current_time - single_team_start_time[0]
                                remaining_team = list(running_teams)[0]
                                
                                if single_team_elapsed > 60:  # 60 seconds timeout for single team
                                    print(f"    LAST TEAM TIMEOUT: Team {remaining_team} stuck for {single_team_elapsed:.1f}s")
                                    print(f"      Terminating stuck team after 60-second timeout...")
                                    
                                    # Find and cancel the future for this team
                                    for future, team_id in future_to_team.items():
                                        if team_id == remaining_team and not future.done():
                                            print(f"       Cancelling stuck Team {team_id}...")
                                            future.cancel()
                                            running_teams.discard(team_id)
                                            # Add timeout error result
                                            with self._results_lock:
                                                self.banner_results[team_id] = {"error": "Last team timeout after 60 seconds"}
                                            break
                                    
                                    print(f"    Terminated last stuck team")
                                    termination_requested.set()  # 🔧 ADD: Signal main thread to stop waiting
                                    break
                                else:
                                    print(f"       Single team running for {single_team_elapsed:.1f}s (timeout in {60-single_team_elapsed:.1f}s)")
                            
                            # Warning if teams are taking too long
                            if elapsed > 300:  # 5 minutes
                                print(f"    WARNING: Teams running for {elapsed/60:.1f} minutes")
                                print(f"      Teams {sorted(list(running_teams))} may be stuck or processing complex images")
                                    
                        else:
                            print(f"    All teams completed!")
                            break
                        
                        # Update count for next iteration
                        last_running_count[0] = current_running_count
                
                # Start monitoring thread
                monitor_thread = threading.Thread(target=periodic_status_monitor, daemon=True)
                monitor_thread.start()
                
                # 🔧 FIXED: Better result collection with termination signal handling
                collected_teams = set()  # Track which teams we've collected results from
                
                # Collect results with periodic termination checks
                while len(collected_teams) < len(future_to_team) and not termination_requested.is_set():
                    try:
                        # Use shorter timeout and check termination signal frequently
                        for future in concurrent.futures.as_completed(future_to_team, timeout=10):  # 10 second chunks
                            team_id = future_to_team[future]
                            
                            # 🔧 PREVENT DUPLICATES: Skip if already collected
                            if team_id in collected_teams:
                                continue
                                
                            collected_teams.add(team_id)
                            
                            try:
                                if future.cancelled():
                                    print(f" Team {team_id} was cancelled (stuck team)")
                                    completed_teams.append((team_id, "cancelled"))
                                    running_teams.discard(team_id)
                                else:
                                    result_team_id, status = future.result(timeout=10)  # 10 second timeout for getting result
                                    completed_teams.append((result_team_id, status))
                                    running_teams.discard(result_team_id)
                                    
                                    print(f" Team {result_team_id} finished: {status}")
                                
                                print(f" Progress: {len(completed_teams)}/{self.num_teams} teams completed")
                                
                                if running_teams:
                                    print(f" Still running: Teams {sorted(list(running_teams))}")
                                else:
                                    print(f" All teams completed!")
                                    break  # Exit early if all teams are done
                                    
                            except concurrent.futures.TimeoutError:
                                print(f" Team {team_id} timed out")
                                running_teams.discard(team_id)
                                with self._results_lock:
                                    self.banner_results[team_id] = {"error": "Execution timeout"}
                            except Exception as exc:
                                print(f" Team {team_id} generated an exception: {exc}")
                                running_teams.discard(team_id)
                                with self._results_lock:
                                    self.banner_results[team_id] = {"error": str(exc)}
                            
                            # Break inner loop if termination requested
                            if termination_requested.is_set():
                                print(" Termination requested by monitoring thread, stopping collection...")
                                break
                                
                        # Break outer loop if we've got all teams or termination was requested
                        if len(collected_teams) >= len(future_to_team) or termination_requested.is_set():
                            break
                            
                    except concurrent.futures.TimeoutError:
                        # This is expected every 10 seconds, just continue and check termination signal
                        if termination_requested.is_set():
                            print(" Termination requested during timeout, breaking...")
                            break
                        continue
                
                # Handle any teams that weren't collected due to cancellation or timeout
                missing_teams = set(self.selected_team_ids) - collected_teams
                if missing_teams:
                    print(f" Handling {len(missing_teams)} missing teams: {missing_teams}")
                    for missing_team in missing_teams:
                        print(f" Team {missing_team} not collected, likely cancelled or timed out")
                        completed_teams.append((missing_team, "timeout_not_collected"))
                        running_teams.discard(missing_team)
                        with self._results_lock:
                            if missing_team not in self.banner_results:
                                self.banner_results[missing_team] = {"error": "Team not collected, likely cancelled"}
                
                
                
                print(f"   Collection phase complete: {len(completed_teams)} teams collected")
                
                #  ENHANCED: Final cleanup with detailed status reporting
                print(f"   Final cleanup: Ensuring all threads are completed...")
                remaining_futures = [future for future in future_to_team if not future.done()]
                
                if remaining_futures:
                    print(f" {len(remaining_futures)} futures still running. Attempting graceful completion...")
                    for i, future in enumerate(remaining_futures, 1):
                        team_id = future_to_team[future]
                        try:
                            print(f"   {i}/{len(remaining_futures)}: Waiting for Team {team_id}...")
                            future.result(timeout=30)  # Extended timeout for cleanup
                            print(f"    Team {team_id} completed during cleanup")
                        except concurrent.futures.TimeoutError:
                            print(f"    Team {team_id} still running after cleanup timeout")
                        except Exception as e:
                            print(f"    Team {team_id} failed during cleanup: {e}")
                else:
                    print(f" All futures completed normally")
                        
        except concurrent.futures.TimeoutError:
            print(" Overall execution timeout - some teams may not have completed")
        except Exception as e:
            print(f" Error in concurrent execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            #  STOP MONITORING: Clean up the monitoring thread
            if 'monitoring_active' in locals():
                monitoring_active.clear()
                print(f" Stopped background monitoring thread")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n Concurrent generation complete!")
        print(f" Total execution time: {total_time:.2f}s")
        print(f" Generated {len(self.banner_results)} banners concurrently")
        print(f" Teams completed: {len(completed_teams)}")
        print(f" Persistent teams created: {len(self.active_teams)}")
        
        # Show completion summary
        success_count = sum(1 for result in self.banner_results.values() if "error" not in result)
        error_count = len(self.banner_results) - success_count
        print(f" Successful: {success_count},  Errors: {error_count}")
        
        # Ensure we have results for all selected teams (add empty results for missing ones)
        for team_id in self.selected_team_ids:
            if team_id not in self.banner_results:
                print(f" No result for Team {team_id}, adding empty result")
                self.banner_results[team_id] = {"error": "No result generated"}
        
        print(" Ready to proceed to judge evaluation...")

    def _generate_initial_banners(self, item_description):
        """
        DEPRECATED: Sequential version kept for reference
        Generate initial banners using configurable number of parallel teams
        """
        print(f" Round 1: Initial Banner Generation ({self.num_teams} sequential teams) - DEPRECATED")
        
        # Get competitive prompt template
        prompt_template = CompetitivePrompt.getCompetitivePrompts("AD Banner Generation", item_description, self.llm, selected_team_ids=self.selected_team_ids)
        
        # Generate banners for each selected team (SEQUENTIAL - OLD METHOD)
        for team_id in self.selected_team_ids:
            print(f"\n Generating Banner #{team_id}...")
            
            # Create individual team prompt
            team_key = f"BannerTeam{team_id}"
            team_config = prompt_template["team"][team_key]
            
            # Build the team
            react_generator = agent_team.ReactAgent(
                intermediate_output_desc=prompt_template["intermediate_output_desc"], 
                config=self.config, 
                llm=self.llm
            )
            
            team = agent_team.buildTeam(
                team_config, 
                react_generator, 
                prompt_template["intermediate_output_desc"], 
                prompt_template["int_out_format"]
            )
            
            # Execute team
            try:
                team_prompt = team_config["prompt"].replace("{item}", item_description)
                
                # Create background message for the banner team
                background_content = f"Generate AD Banner for: {item_description}. Design Philosophy: {team_config['additional_prompt']}. This is part of a competitive banner generation workflow where {self.num_teams} teams create different design approaches."
                
                result = team(
                    state={
                        "history": {
                            f"BannerGeneration{team_id} Supervisor": [HumanMessage(content=team_prompt)], 
                            "all": [HumanMessage(content=team_prompt)]
                        }, 
                        "intermediate_output": {},
                        "recent_files": {},  # Add file tracking support
                        "background": AIMessage(content=background_content)
                    },
                    config={"recursion_limit": int(self.config["LLM"]["max_attempts"])}
                )
                
                # Store result
                if isinstance(result, list):
                    self.banner_results[team_id] = result[0]["intermediate_output"]
                else:
                    self.banner_results[team_id] = result["intermediate_output"]
                    
                print(f" Banner #{team_id} completed")
                
            except Exception as e:
                print(f" Error generating Banner #{team_id}: {e}")
                import traceback
                traceback.print_exc()
                self.banner_results[team_id] = {"error": str(e)}
        
        print(f"\n Initial generation complete. Generated {len(self.banner_results)} banners.")
    
    def _execute_judge_evaluation(self):
        """
        Execute Judge Agent evaluation of active banners
        """
        print(f"\n Judge Agent evaluating {len(self.active_banners)} active banners...")
        
        # Prepare banner data for judge AND extract image paths
        active_banner_data = {}
        recent_files = {}
        
        print(f" Judge Agent starting to collect final images from active banners...")
        print(f" Active banners: {self.active_banners}")
        print(f" Current round: {self.round_number}")
        
        for banner_id in self.active_banners:
            if banner_id in self.banner_results and "error" not in self.banner_results[banner_id]:
                banner_data = self.banner_results[banner_id]
                active_banner_data[banner_id] = banner_data
                
                #  Find this team's final image in the current round (highest revision number)
                team_final_image = self._find_team_final_image(banner_id, self.round_number)
                
                if team_final_image:
                    #  Use original key name format to avoid duplication
                    recent_files[f"banner_{banner_id}_image"] = team_final_image
                    print(f" Team {banner_id}: {os.path.basename(team_final_image)}")
                else:
                    print(f" Team {banner_id}: Final image not found")
        
        #  Verify collection results
        unique_images = list(set(recent_files.values()))  # Remove duplicates
        print(f"\n Collection result verification:")
        print(f"   Active banners: {len(self.active_banners)}")
        print(f"   Recent files entries: {len(recent_files)}")
        print(f"   Unique images: {len(unique_images)}")
        
        if len(unique_images) == len(active_banner_data):
            print(f" Collection correct: Each active team has one final image")
        else:
            print(f" Collection abnormal: Image count doesn't match team count")
        
        print(f" Judge Agent will analyze {len(unique_images)} final images:")
        for i, img_path in enumerate(unique_images, 1):
            print(f"   {i}. {os.path.basename(img_path)}")
        
        print(f" Recent files keys: {list(recent_files.keys())}")
        
        # Create judge prompt that emphasizes these are the FINAL images of each team
        banner_data_str = ""
        for banner_id, data in active_banner_data.items():
            banner_data_str += f"\nBanner #{banner_id} (Team {banner_id} - Round {self.round_number} FINAL):\n"
            if isinstance(data, dict):
                for key, value in data.items():
                    # If this looks like an image path, emphasize it
                    if isinstance(value, str) and any(ext in value.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                        banner_data_str += f"  - {key}: {value}  [IMAGE FILE - USE image_to_base64 TOOL]\n"
                    else:
                        banner_data_str += f"  - {key}: {value}\n"
            else:
                banner_data_str += f"  - {str(data)}\n"
            
            # Add final image path information
            if f"banner_{banner_id}_image" in recent_files:
                final_image_path = recent_files[f"banner_{banner_id}_image"]
                banner_data_str += f"  - FINAL_IMAGE_PATH: {final_image_path} [TEAM'S FINAL IMAGE - ANALYZE THIS]\n"
        
        judge_prompt = f"""
        Evaluate and rank the following {len(active_banner_data)} AD banners based on their FINAL images:
        
        EVALUATION TASK SUMMARY:
        - Round Number: {self.round_number}
        - Active Banners: {list(active_banner_data.keys())}
        - Total Images to Analyze: {len(unique_images)}
        
        CRITICAL: These are the FINAL images from each team after {self.max_revisions_per_team} possible revisions.
        Each team's best work is represented by their highest revision number image.
        
        AVAILABLE IMAGE FILES:
        {chr(10).join([f"   - Team {banner_id}: {recent_files[f'banner_{banner_id}_image']}" for banner_id in active_banner_data.keys() if f'banner_{banner_id}_image' in recent_files])}
        
        Banner Data:{banner_data_str}
        
        CRITICAL INSTRUCTIONS FOR IMAGE ANALYSIS:
        1. You MUST use the image_to_base64 tool to analyze EACH team's FINAL banner image 
        2. Analyze ALL {len(unique_images)} images separately - each team's FINAL_IMAGE_PATH
        3. File naming format: generated_image_ImageResearcher_Team{{ID}}_Round{self.round_number}_{{MaxRevision}}.png
        4. For each image, analyze: colors, layout, text readability, brand consistency, visual appeal
        5. Score each banner based on what you SEE in the FINAL images, not just the metadata
        6. Compare all {len(active_banner_data)} banners fairly and rank them from best to worst
        
        REQUIRED OUTPUT:
        Please analyze each team's FINAL image and provide detailed evaluation and ranking.
        Since you have {len(active_banner_data)} active banners, you must rank all of them.
        {"If more than 1 banner remains, identify the worst performer for elimination." if len(active_banner_data) > 1 else "Since only 1 banner remains, declare it as the winner."}
        
        Please provide your evaluation in the required JSON format as intermediate_output:
        {{
          "round_number": {self.round_number},
          "active_banners": {list(active_banner_data.keys())},
          "rankings": [best_to_worst_banner_ids],
          {"\"worst_banner\": worst_banner_id," if len(active_banner_data) > 1 else "\"winner\": winner_banner_id,"}
          "elimination_reason": "detailed explanation based on FINAL image visual analysis",
          {chr(10).join([f'          "banner_{banner_id}_evaluation": {{"scores": {{"visual_appeal": score, "brand_consistency": score, "message_clarity": score, "technical_quality": score}}, "total_score": total, "strengths": ["strength1", "strength2", "strength3"], "weaknesses": ["weakness1", "weakness2", "weakness3"], "improvements": ["suggestion1", "suggestion2", "suggestion3"]}},' for banner_id in active_banner_data.keys()])}
          "comparative_analysis": "detailed cross-banner insights based on FINAL image visual inspection of all {len(active_banner_data)} banners"
        }}
        """
        
        # Execute Multi-Judge Voting System
        try:
            # Get prompt template with all judge configurations
            prompt_template = CompetitivePrompt.getCompetitivePrompts("AD Banner Generation", "", self.llm, selected_team_ids=None, config=self.config)
            
            print(f" Starting Multi-Judge Voting System with 5 specialized judges...")
            
            # List of all specialized judges
            possible_judge_types = [
                ("VisualDesignJudge", "Visual Design"),
                ("CopywritingJudge", "Copywriting & Marketing"), 
                ("BrandConsistencyJudge", "Brand Consistency"),
                ("UserExperienceJudge", "User Experience"),
                ("TechnicalQualityJudge", "Technical Quality")
            ]
            
            if self.judge_combination == "all":
                judge_types = possible_judge_types
            elif self.judge_combination == "1":
                judge_types = [possible_judge_types[0]]
            elif self.judge_combination == "2":
                judge_types = [possible_judge_types[1]]
            elif self.judge_combination == "3":
                judge_types = [possible_judge_types[2]]
            elif self.judge_combination == "4":
                judge_types = [possible_judge_types[3]]
            elif self.judge_combination == "5":
                judge_types = [possible_judge_types[4]]
            elif self.judge_combination == "1,2":
                judge_types = [possible_judge_types[0], possible_judge_types[1]]
            elif self.judge_combination == "1,2,3":
                judge_types = [possible_judge_types[0], possible_judge_types[1], possible_judge_types[2]]
            elif self.judge_combination == "1,2,3,4":
                judge_types = [possible_judge_types[0], possible_judge_types[1], possible_judge_types[2], possible_judge_types[3]]
            elif self.judge_combination == "1,2,3,4,5":
                judge_types = possible_judge_types  # All 5 judges
            else:
                # Default fallback: use all judges
                print(f"⚠️ Unknown judge_combination '{self.judge_combination}', using all judges")
                judge_types = possible_judge_types
            all_judge_votes = {}
            
            # Execute each specialized judge in parallel
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(judge_types)) as executor:
                future_to_judge = {}
                
                for judge_key, judge_name in judge_types:
                    # Create individual judge team configuration
                    judge_team_config = {
                        "team": f"{judge_key}Evaluation",
                        "return": "FINISH", 
                        "prompt": f"Specialized {judge_name} evaluation of AD banners for competitive selection.",
                        "additional_prompt": f"Focus on {judge_name.lower()} aspects and vote RECOMMEND or ELIMINATE for each banner.",
                        judge_key: prompt_template["team"][judge_key]
                    }
                    
                    # Create judge-specific prompt
                    judge_specific_prompt = f"""
                    Specialized {judge_name} Judge Evaluation:
                    
                    {judge_prompt}
                    
                    As a {judge_name} specialist, evaluate each banner and vote RECOMMEND or ELIMINATE.
                    Focus specifically on {judge_name.lower()} aspects while analyzing the banner images.
                    """
                    
                    # Submit judge execution to thread pool
                    future = executor.submit(self._execute_single_judge, 
                                           judge_team_config, judge_specific_prompt, 
                                           recent_files, prompt_template, judge_key, judge_name)
                    future_to_judge[future] = (judge_key, judge_name)
                
                # Collect all judge votes
                for future in concurrent.futures.as_completed(future_to_judge, timeout=300):  # 5 minute timeout
                    judge_key, judge_name = future_to_judge[future]
                    try:
                        judge_vote = future.result(timeout=60)  # 1 minute timeout per judge
                        all_judge_votes[judge_key] = judge_vote
                        print(f" {judge_name} Judge vote completed")
                    except Exception as exc:
                        print(f" {judge_name} Judge failed: {exc}")
                        # Fallback vote if judge fails
                        all_judge_votes[judge_key] = {
                            "judge_type": judge_key.lower(),
                            "banner_votes": {f"banner_{bid}": {"vote": "RECOMMEND", "reasoning": "Judge failed, default recommend"} 
                                           for bid in active_banner_data.keys()},
                            "error": str(exc)
                        }
            
            print(f" All judge votes collected. Processing voting results...")
            
            # Process voting results with VotingCoordinator
            voting_result = self._process_multi_judge_votes(all_judge_votes, active_banner_data, prompt_template, recent_files)
            
            return voting_result
            
        except Exception as e:
            print(f" Error in multi-judge evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: random elimination if judge fails
            return {
                "round_number": self.round_number,
                "active_banners": self.active_banners,
                "worst_banner": self.active_banners[-1] if len(self.active_banners) > 1 else None,
                "elimination_reason": f"Multi-judge evaluation failed: {e}",
                "rankings": self.active_banners
            }

    def _find_team_final_image(self, team_id, round_number):
        """
        Find the final image (highest revision number) for a specific team and round
        Expected format: generated_image_ImageResearcher_Team{ID}_Round{R}_{Rev}.png
        Returns the path to the image with the highest revision number
        """
        import glob
        output_dir = self.config.get("SETTING", "output_folder", fallback="outputs")
        
        # Search pattern for this team's images in the current round
        pattern = f"{output_dir}/generated_image_*_Team{team_id}_Round{round_number}_*.png"
        
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # Fallback: try alternative patterns
            fallback_patterns = [
                f"{output_dir}/generated_image_*Team{team_id}*Round{round_number}*.png",
                f"{output_dir}/*Team{team_id}*Round{round_number}*.png",
                f"{output_dir}/generated_image_*Team{team_id}*.png"
            ]
            
            for fallback_pattern in fallback_patterns:
                matching_files.extend(glob.glob(fallback_pattern))
                if matching_files:
                    break
        
        if not matching_files:
            print(f" No images found for Team {team_id} Round {round_number} with pattern: {pattern}")
            return None
        
        # Find the image with the highest revision number
        final_image = None
        max_revision = -1
        
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            try:
                # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
                parts = filename.replace('.png', '').split('_')
                if len(parts) >= 6:
                    team_part = parts[3]  # Team1
                    round_part = parts[4]  # Round1  
                    revision_part = parts[5]  # 3
                    
                    # Verify this is the correct team and round
                    if (team_part == f"Team{team_id}" and 
                        round_part == f"Round{round_number}"):
                        revision_num = int(revision_part)
                        if revision_num > max_revision:
                            max_revision = revision_num
                            final_image = file_path
            except Exception as e:
                print(f"Could not parse filename: {filename}, error: {e}")
                continue
        
        if final_image:
            print(f"Found Team {team_id} final image: {os.path.basename(final_image)} (revision {max_revision})")
            return final_image
        else:
            # Fallback: return the most recent file by modification time
            most_recent = max(matching_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
            print(f"Using most recent file for Team {team_id}: {os.path.basename(most_recent)}")
            return most_recent
    
    def _process_elimination(self, judge_result):
        """
        Process the elimination based on judge results
        Removes eliminated teams from active_teams to free resources
        """
        try:
            if "worst_banner" in judge_result:
              

                eliminated_banner_key = judge_result["worst_banner"]
                elimination_reason = judge_result.get("elimination_reason", "Judge decision")
                
                # 🎯 FIX: Map banner_X_image format back to team ID
                # Judge Agent returns keys like "banner_2_image", but active_banners contains team IDs like 2
                eliminated_team_id = None
                if type(judge_result["worst_banner"]) == int:
                    eliminated_banner_key = f"banner_{judge_result['worst_banner']}_image"
                    eliminated_team_id = judge_result["worst_banner"]
                else:
                    # Method 1: Extract team ID from banner_X_image format
                    if eliminated_banner_key.startswith("banner_") and eliminated_banner_key.endswith("_image"):
                        try:
                            # Extract the number from "banner_X_image"
                            team_id_str = eliminated_banner_key.replace("banner_", "").replace("_image", "")
                            eliminated_team_id = int(team_id_str)
                            print(f"Mapped '{eliminated_banner_key}' to Team ID {eliminated_team_id}")
                        except ValueError:
                            print(f"Could not extract team ID from '{eliminated_banner_key}'")
                    
                    # Method 2: Fallback - check if eliminated_banner_key is directly in active_banners (for backwards compatibility)
                    if eliminated_team_id is None and eliminated_banner_key in self.active_banners:
                        eliminated_team_id = eliminated_banner_key
                        print(f"Direct match found: {eliminated_banner_key}")
                    
                    # Method 3: Final fallback - try to find by matching patterns
                    if eliminated_team_id is None:
                        print(f"Could not map '{eliminated_banner_key}' to any active team")
                        print(f"Active banners: {self.active_banners}")
                        print(f"Attempting pattern matching...")
                        
                        # Look for any number in the eliminated_banner_key that matches an active banner
                        import re
                        numbers_in_key = re.findall(r'\d+', eliminated_banner_key)
                        for num_str in numbers_in_key:
                            try:
                                potential_id = int(num_str)
                                if potential_id in self.active_banners:
                                    eliminated_team_id = potential_id
                                    print(f"Pattern match found: {eliminated_banner_key} -> Team {eliminated_team_id}")
                                    break
                            except ValueError:
                                continue
                
                if eliminated_team_id is not None and eliminated_team_id in self.active_banners:
                    # Remove from active banners list
                    self.active_banners.remove(eliminated_team_id)
                    
                    # 🧠 REMOVE ELIMINATED TEAM FROM ACTIVE TEAMS (free resources)
                    if eliminated_team_id in self.active_teams:
                        del self.active_teams[eliminated_team_id]
                        print(f"Team #{eliminated_team_id} removed from active teams pool")
                        
                    if eliminated_team_id in self.team_states:
                        del self.team_states[eliminated_team_id]
                        print(f"Team #{eliminated_team_id} state cleared from memory")
                    
                    # Record elimination history
                    self.elimination_history.append({
                        "round": self.round_number,
                        "eliminated": eliminated_team_id,
                        "reason": elimination_reason,
                        "remaining": self.active_banners.copy()
                    })
                    
                    print(f" Banner #{eliminated_team_id} eliminated (was: {eliminated_banner_key})")
                    print(f" Reason: {elimination_reason}")
                    print(f" Remaining: {self.active_banners}")
                    print(f" Active teams in memory: {list(self.active_teams.keys())}")
                
                else:
                    print(f"No valid elimination decision from judge")
                    print(f"   Judge returned: '{eliminated_banner_key}'")
                    print(f"   Mapped to: {eliminated_team_id}")
                    print(f"   Active banners: {self.active_banners}")
                    print(f"   Could not find match for elimination")
                    
        except Exception as e:
            print(f"Error processing elimination: {e}")
            import traceback
            traceback.print_exc()
    
    def _improve_remaining_banners(self, judge_result):
        """
        Improve remaining banners based on judge feedback
        Uses EXISTING team instances with their preserved memory and state
        Adds judge feedback to teams' conversation history
        """
        print(f"\nImproving remaining {len(self.active_banners)} banners based on judge feedback...")
        
        if len(self.active_banners) <= 1:
            print("Only one banner remaining, no improvements needed")
            return
        
        # ✅ 正确时机：在judge评价完成后，开始improvement前递增round
        previous_round = self.round_number
        self.round_number += 1
        current_round_number = self.round_number
        
        print(f"ROUND TRANSITION: {previous_round} → {current_round_number}")
        print(f"All teams will now generate Round {current_round_number} images")
        
        # Get the elimination reason and comparative analysis for context
        elimination_reason = judge_result.get("elimination_reason", "")
        comparative_analysis = judge_result.get("comparative_analysis", "")
        eliminated_banner = judge_result.get("worst_banner", "")
        
        print(f"Using PERSISTENT team instances (no recreation needed)")
        print(f"Adding Judge feedback to existing team conversations...")
        
        # CRITICAL: Update all active teams' tools to current round BEFORE improvement
        print(f"Updating all active teams from Round {previous_round} to Round {current_round_number}")
        tools_updated_total = 0
        for banner_id in self.active_banners:
            if banner_id in self.active_teams:
                team = self.active_teams[banner_id]
                tools_updated = self._update_team_tools_round_number(team, current_round_number)
                tools_updated_total += tools_updated
                print(f" Team {banner_id}: Updated to Round {current_round_number}")
            else:
                print(f" Team {banner_id} not found in active_teams")
        
        print(f"ROUND UPDATE COMPLETE: {tools_updated_total} tools updated across {len(self.active_banners)} teams")
        
        # Use ThreadPoolExecutor for concurrent banner improvements
        improvement_start_time = time.time()
        completed_improvements = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.active_banners)) as executor:
                # Submit improvement tasks for all remaining banners using EXISTING teams
                future_to_banner = {}
                
                for banner_id in self.active_banners:
                    if banner_id in self.active_teams:  # Only improve teams that exist
                        future = executor.submit(self._continue_with_existing_team, banner_id, judge_result, elimination_reason, comparative_analysis)
                        future_to_banner[future] = banner_id
                    else:
                        print(f" Team {banner_id} not found in active_teams, skipping")
                
                # Collect improvement results
                for future in concurrent.futures.as_completed(future_to_banner, timeout=600):  # 10 minute timeout
                    banner_id = future_to_banner[future]
                    try:
                        result_data = future.result(timeout=10)
                        completed_improvements.append((result_data["banner_id"], "success"))
                        print(f"Banner #{result_data['banner_id']} improvement: success")
                    except concurrent.futures.TimeoutError:
                        print(f"Banner #{banner_id} improvement timed out")
                        # Keep the original banner if improvement fails
                    except Exception as exc:
                        print(f"Banner #{banner_id} improvement failed: {exc}")
                        # Keep the original banner if improvement fails
        
        except Exception as e:
            print(f"Error in concurrent banner improvement: {e}")
        
        improvement_end_time = time.time()
        improvement_time = improvement_end_time - improvement_start_time
        
        print(f"Banner improvement complete!")
        print(f"Total improvement time: {improvement_time:.2f}s")
        print(f"Improved banners: {len(completed_improvements)}")
        print(f"Teams maintained their conversation history and memory")
        print(f"Ready for next round evaluation...")

    def _continue_with_existing_team(self, banner_id, judge_result, elimination_reason, comparative_analysis):
        """
        Continue with existing team instance, adding judge feedback to conversation history
        This preserves the team's memory and context while providing new improvement instructions
        """
        try:
            print(f"\n[Continuation] Using existing Banner #{banner_id} team...")
            
            # Get the existing team instance and current state
            team = self.active_teams[banner_id]
            current_state = self.team_states[banner_id]
            
            # 🔧 COMPREHENSIVE STATE VALIDATION AND REPAIR
            print(f"[Team {banner_id}] Validating state structure...")
            
            # 1. Fix messages field
            if "messages" not in current_state:
                current_state["messages"] = []
                print(f"[Team {banner_id}] Added missing messages field")
            elif not isinstance(current_state["messages"], list):
                # If messages is a single message, convert to list
                if hasattr(current_state["messages"], '__iter__') and not isinstance(current_state["messages"], str):
                    current_state["messages"] = list(current_state["messages"])
                else:
                    current_state["messages"] = [current_state["messages"]]
                print(f" [Team {banner_id}] Converted messages to list")
            
            # 2.  CRITICAL FIX: Ensure background field always exists and is AIMessage
            if "background" not in current_state or current_state["background"] is None:
                # Create proper background content for improvement round
                background_content = f"""ROUND {current_round_number} IMPROVEMENT TASK: 
Banner #{banner_id} enhancement based on judge feedback. 
Design Philosophy: {self.design_philosophies.get(banner_id, 'Unknown')}. 
This is Round {current_round_number} with FRESH revision limits. 
Previous round limits are cleared. Your team ID is {banner_id}. 
"""
                current_state["background"] = AIMessage(content=background_content)
                print(f" [Team {banner_id}] Updated background for Round {current_round_number}")
            elif not hasattr(current_state["background"], "content"):
                # Background exists but wrong type - convert to AIMessage
                background_content = str(current_state["background"])
                current_state["background"] = AIMessage(content=background_content)
                print(f"[Team {banner_id}] Converted background to AIMessage type")
            
            # 3. Fix other essential state fields
            if "history" not in current_state:
                current_state["history"] = {f"BannerGeneration{banner_id} Supervisor": []}
                print(f"[Team {banner_id}] Added missing history field")
            
            if "intermediate_output" not in current_state:
                current_state["intermediate_output"] = {}
                print(f"[Team {banner_id}] Added missing intermediate_output field")
                
            if "recent_files" not in current_state:
                current_state["recent_files"] = {}
                print(f"[Team {banner_id}] Added missing recent_files field")
            
            # 4. Ensure team_id and round_number are set
            current_state["team_id"] = banner_id
            current_state["round_number"] = self.round_number
            print(f"\n[TARGETED FIX] Removing GraphicRevisor hard limit message")

            # 1. 专门清理GraphicRevisor的hard limit消息
            if "GraphicRevisor" in current_state["history"]:
                graphic_history = current_state["history"]["GraphicRevisor"]
                clean_graphic_history = []
                
                for msg in graphic_history:
                    if hasattr(msg, 'content'):
                        content = str(msg.content).lower()
                        if not any(keyword in content for keyword in ['hard limit', 'no further revisions', 'limit reached']):
                            clean_graphic_history.append(msg)
                
                current_state["history"]["GraphicRevisor"] = clean_graphic_history
                print(f"GraphicRevisor: {len(graphic_history)} → {len(clean_graphic_history)} messages")

            # 2. 强制清理intermediate_output并验证
            current_state["intermediate_output"] = {"status": "ready_for_round_2", "round_number": self.round_number}
            print(f"Forced intermediate_output reset: {current_state['intermediate_output']}")

            # 3. 添加强制性的Round 2指令到Supervisor history
            supervisor_key = f"BannerGeneration{banner_id} Supervisor"
            round2_instruction = HumanMessage(content=f"""
            ROUND {self.round_number} STARTED - The limitation of how many times you can revise your banner image is cleared.

            Critical Update: You are now in Round {self.round_number}. 
            Any previous "hard limit reached" messages regarding revision limitations for the banner imagesfrom Round 1 are not related to this round.
            Your revision limits have been RESET for this new round.

            Judge feedback requires you to generate an IMPROVED banner.
            You MUST create a improved Round {self.round_number} image based on the last image you generated and the feedback from the previous round.
            """)

            current_state["history"][supervisor_key].append(round2_instruction)
            print(f"Added Round {self.round_number} reset instruction to Supervisor")
            # 5. Validation summary
            print(f"[Team {banner_id}] State validation complete:")
            print(f"    messages: {type(current_state.get('messages', 'MISSING'))} (length: {len(current_state.get('messages', []))})")
            print(f"    background: {type(current_state.get('background', 'MISSING'))}")
            print(f"    history: {type(current_state.get('history', 'MISSING'))}")
            print(f"    team_id: {current_state.get('team_id', 'MISSING')}")
            print(f"    round_number: {current_state.get('round_number', 'MISSING')}")
            
            # Verify background can be accessed safely (what supervisor_agent needs)
            try:
                _ = current_state["background"].content
                print(f"    background.content access: OK")
            except Exception as e:
                print(f"    background.content access failed: {e}")
                # Last resort fix
                current_state["background"] = AIMessage(content=f"Team {banner_id} improvement task")
                print(f"    Applied emergency background fix")
            
            # ✅ 使用已经更新的当前round_number（在_improve_remaining_banners中已更新）
            current_round_number = self.round_number
            print(f"Using current round_number={current_round_number} for Team {banner_id}")
            
            # Extract specific feedback for this banner
            feedback_key = f"banner_{banner_id}_evaluation"
            feedback = judge_result.get(feedback_key, {})
            
            # FIX: Also try alternative feedback key formats since Judge Agent might use different naming
            if not feedback:
                # Try alternative formats that Judge Agent might use
                alternative_keys = [
                    f"banner_{banner_id}_image_evaluation",  # If Judge includes "_image" in key
                    f"banner{banner_id}_evaluation",         # Without underscore
                    f"team_{banner_id}_evaluation",          # If Judge uses "team" prefix
                    f"team{banner_id}_evaluation"            # Without underscore
                ]
                for alt_key in alternative_keys:
                    if alt_key in judge_result:
                        feedback = judge_result[alt_key]
                        print(f"Found feedback using alternative key: {alt_key}")
                        break
            
            if not feedback:
                print(f"No specific feedback found for Banner #{banner_id}")
                feedback = {"improvements": ["General improvements needed based on judge evaluation"]}
            
            # 🎯 ENHANCED: Create more explicit improvement prompt that demands new image generation
            # 🎯 BANNER SIZE ENFORCEMENT: Get the correct banner size from config
            banner_size = "1024x1024"  # Default fallback
            try:
                if hasattr(self.config, 'get') and hasattr(self.config, 'has_section'):
                    # ConfigParser object
                    if self.config.has_section("BANNER_CONFIG") and self.config.has_option("BANNER_CONFIG", "banner_size"):
                        banner_size = self.config.get("BANNER_CONFIG", "banner_size")
                elif isinstance(self.config, dict):
                    # Dictionary object
                    if "BANNER_CONFIG" in self.config and "banner_size" in self.config["BANNER_CONFIG"]:
                        banner_size = self.config["BANNER_CONFIG"]["banner_size"]
            except Exception as e:
                print(f"⚠️ Warning: Unable to read banner_size from config, using default 1024x1024: {e}")
            
            print(f"🎯 [Round {current_round_number}] Banner size enforced in improvement prompt: {banner_size}")
            
            # 🎯 LOGO PATH: Hardcode logo path based on output directory
            # Logo files are copied to output directory with original filename
            output_folder = self.config.get("SETTING", "output_folder") if hasattr(self.config, 'get') else "../output"
            
            # Find logo file in output directory (pattern: *.png that's not generated_image_*)
            import glob
            logo_files = glob.glob(f"{output_folder}/*.png")
            logo_path = None
            for file in logo_files:
                if not os.path.basename(file).startswith('generated_image_'):
                    logo_path = file
                    break
            
            last_banner_path = self._find_team_final_image(banner_id, current_round_number - 1)
            
            print(f"🔍 [Round {current_round_number}] Debug dual input setup:")
            print(f"   output_folder: {output_folder}")
            print(f"   found logo_path: {logo_path}")
            print(f"   logo_path exists: {os.path.exists(logo_path) if logo_path else False}")
            print(f"   last_banner_path: {last_banner_path}")
            print(f"   last_banner_path exists: {os.path.exists(last_banner_path) if last_banner_path else False}")
            
            # Prepare input files instruction for revision: [previous_banner, original_logo]
            input_files_instruction = ""
            if last_banner_path and logo_path and os.path.exists(last_banner_path) and os.path.exists(logo_path):
                input_files_instruction = f"""
🖼️ DUAL INPUT REQUIREMENT: You MUST use BOTH images as input for revision:
1. Previous banner: {os.path.basename(last_banner_path)} (your Round {current_round_number - 1} result)
2. Original logo: {os.path.basename(logo_path)} (to ensure logo consistency)

Input format for gemini_image_generator: input_filepath="[\\"{last_banner_path}\\", \\"{logo_path}\\"]"

CRITICAL: This dual-input approach ensures the logo remains consistent while improving the banner design.
The first image is your previous banner to be improved, the second image is the original logo to maintain consistency.
"""
                print(f"🖼️ [Round {current_round_number}] Dual input setup:")
                print(f"   Previous banner: {last_banner_path}")
                print(f"   Original logo: {logo_path}")
            elif last_banner_path and os.path.exists(last_banner_path):
                input_files_instruction = f"""
🖼️ INPUT REQUIREMENT: Use your previous banner as input for revision:
Previous banner: {os.path.basename(last_banner_path)} (your Round {current_round_number - 1} result)

Input format for gemini_image_generator: input_filepath="[\\"{last_banner_path}\\"]"

⚠️ CRITICAL: Use the EXACT path above, not variable names like 'most_recent_image_filepath'
"""
                print(f"🖼️ [Round {current_round_number}] Single input setup: {last_banner_path}")
            elif logo_path and os.path.exists(logo_path):
                input_files_instruction = f"""
🖼️ LOGO ONLY INPUT: Previous banner not found, use original logo as reference:
Original logo: {os.path.basename(logo_path)}

Input format for gemini_image_generator: input_filepath="[\\"{logo_path}\\"]"

⚠️ CRITICAL: Use the EXACT path above, not variable names
"""
                print(f"🖼️ [Round {current_round_number}] Logo-only input setup: {logo_path}")
            else:
                input_files_instruction = "⚠️ Neither previous banner nor logo found, generate new banner based on feedback only."
                print(f"⚠️ [Round {current_round_number}] No input files found for Team {banner_id}")
            
            improvement_message = f""" ROUND {current_round_number} IMPROVEMENT TASK
NEW ROUND ALERT: You are now in ROUND {current_round_number} - All previous hard limits regarding revision limitations are cleared!

CRITICAL REQUIREMENT: You MUST generate a revised image for Round {current_round_number} based on the last image you generated. This is NOT optional.

{input_files_instruction}

HARD LIMIT WARNING: You are limited to {self.max_revisions_per_team} total revisions. Image generation tools will be BLOCKED if you exceed this limit.

Judge Feedback for your Round {current_round_number - 1} banner:
{json.dumps(feedback, indent=2)}

Elimination Context: {elimination_reason}

Comparative Analysis: {comparative_analysis}

MANDATORY REQUIREMENTS (NO EXCEPTIONS):
1. You MUST call recraft_image_generator or gemini_image_generator to create a revised image
2. Filename MUST include "Round{current_round_number}" (e.g., generated_image_ImageResearcher_Team{banner_id}_Round{current_round_number}_1.png)
3. Address ALL feedback points mentioned above
4. You CANNOT say "No changes needed" - you MUST make improvements due to hard limit enforcement
5. 🎯 BANNER SIZE REQUIREMENT: You MUST use EXACTLY {banner_size} pixels for all image generation - DO NOT change this size under any circumstances

EXECUTION SEQUENCE (MANDATORY):
1. Read the judge feedback carefully
2. Plan specific improvements to address ALL weaknesses mentioned  
3. ⚠️ CRITICAL: You MUST use gemini_image_generator with the EXACT input_filepath format specified above
   - DO NOT use variable names like 'most_recent_image_filepath' 
   - USE the complete file paths as provided above
   - ALWAYS include both previous banner AND original logo when both are available
4. Generate an improved image file that includes Round{current_round_number} in the name
5. The improved image MUST follow the revision suggestions from Round {current_round_number - 1} version while maintaining logo consistency

You MUST address these specific areas:
- Visual Appeal improvements
- Brand Consistency enhancements  
- Message Clarity optimizations
- Target Audience Fit adjustments
- Technical Quality upgrades
- Innovation additions

HARD LIMIT REMINDER: After {self.max_revisions_per_team} revisions, image generation tools will be automatically BLOCKED.

FAILURE TO GENERATE A NEW IMAGE WILL RESULT IN AUTOMATIC ELIMINATION.

START NOW: Generate your Round {current_round_number} improved banner image using image generation tools with size="{banner_size}" parameter."""

            # Add the improvement message to the team's conversation history
            human_message = HumanMessage(content=improvement_message)
            current_state["messages"].append(human_message)
            
            # Update team state to include the improvement instructions
            self.team_states[banner_id] = current_state
            
            print(f"Added Round {current_round_number} improvement instructions to Banner #{banner_id}")

            # Continue with the existing team instance
            start_time = time.time()
            result = team(current_state, config={
                "recursion_limit": int(self.config["LLM"]["max_attempts"]),
                "team_id": banner_id,
                "improvement_round": True,
                "round_number": current_round_number,  # Force pass round number in config
                "metadata": {  # 🎯 ADD: Include round_number in metadata for tools
                    "round_number": current_round_number,
                    "team_id": banner_id,
                    "workflow_phase": "improvement_round"
                }
            })
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Update the banner results with the improved version
            if isinstance(result, list):
                final_result = result[0]
            else:
                final_result = result
                
            self.banner_results[banner_id] = final_result.get("intermediate_output", {})
            
            # 🔧 FIX: Apply the same state preservation logic as in first round
            preserved_state = {
                "history": final_result.get("history", {}),
                "messages": final_result.get("messages", []),
                "intermediate_output": final_result.get("intermediate_output", {}),
                "recent_files": final_result.get("recent_files", {}),
                "next": final_result.get("next", "FINISH"),
                "team_id": banner_id,
                "round_number": current_round_number
            }
            
            # Ensure background field is preserved
            if "background" in final_result and final_result["background"] is not None:
                if hasattr(final_result["background"], "content"):
                    preserved_state["background"] = final_result["background"]
                else:
                    background_content = str(final_result["background"])
                    preserved_state["background"] = AIMessage(content=background_content)
            else:
                background_content = f"Banner #{banner_id} improvement completed in Round {current_round_number}. Team ID: {banner_id}."
                preserved_state["background"] = AIMessage(content=background_content)
            
            self.team_states[banner_id] = preserved_state
            
            print(f"[Continuation] Banner #{banner_id} improved in {execution_time:.2f}s")
            
            # 🔧 FIX: Handle messages field safely - it could be AIMessage object or list
            messages_info = preserved_state.get('messages', [])
            if hasattr(messages_info, 'content'):  # It's an AIMessage object
                messages_count = 1
            elif isinstance(messages_info, list):  # It's a list of messages
                messages_count = len(messages_info)
            else:
                messages_count = 0
            
            print(f" Conversation history preserved: {messages_count} messages")
            print(f" State structure preserved for next round")
            
            return {
                "banner_id": banner_id,
                "result": final_result,
                "execution_time": execution_time,
                "method": "improved_with_memory",
                "round_number": current_round_number
            }
            
        except Exception as e:
            print(f"[Continuation] Error improving Banner #{banner_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "banner_id": banner_id,
                "result": {"error": str(e)},
                "execution_time": 0,
                "method": "error",
                "round_number": self.round_number
            }
    
    def _execute_single_judge(self, judge_team_config, judge_prompt, recent_files, prompt_template, judge_key, judge_name):
        """
        Execute a single specialized judge evaluation
        """
        try:
            print(f"Executing {judge_name} Judge...")
            
            # Create judge team using buildTeam
            react_generator = agent_team.ReactAgent(
                intermediate_output_desc=prompt_template["intermediate_output_desc"], 
                config=self.config, 
                llm=self.llm
            )
            
            judge_team = agent_team.buildTeam(
                judge_team_config, 
                react_generator, 
                prompt_template["intermediate_output_desc"], 
                prompt_template["int_out_format"]
            )
            
            # Execute judge
            result = judge_team(
                state={
                    "history": {
                        f"{judge_team_config['team']} Supervisor": [HumanMessage(content=judge_prompt)], 
                        "all": [HumanMessage(content=judge_prompt)]
                    }, 
                    "intermediate_output": {},
                    "recent_files": recent_files,
                    "background": AIMessage(content=f"{judge_name} judge evaluation of banner images")
                },
                config={"recursion_limit": int(self.config["LLM"]["max_attempts"])}
            )
            
            # Extract result
            if isinstance(result, list):
                final_result = result[0]["intermediate_output"]
            else:
                final_result = result["intermediate_output"]
            
            print(f"{judge_name} Judge completed evaluation")
            return final_result
            
        except Exception as e:
            print(f"{judge_name} Judge failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _process_multi_judge_votes(self, all_judge_votes, active_banner_data, prompt_template, recent_files):
        """
        Process all judge votes using VotingCoordinator to determine final elimination
        """
        try:
            print(f"Processing votes from {len(all_judge_votes)} judges...")
            
            # Prepare voting summary for VotingCoordinator
            voting_summary_prompt = f"""
            Multi-Judge Voting Results Processing:
            
            Round Number: {self.round_number}
            Active Banners: {list(active_banner_data.keys())}
            
            JUDGE VOTES COLLECTED:
            """
            
            for judge_key, vote_data in all_judge_votes.items():
                voting_summary_prompt += f"\n{judge_key.upper()} VOTE:\n"
                voting_summary_prompt += f"Judge Type: {vote_data.get('judge_type', judge_key)}\n"
                
                banner_votes = vote_data.get('banner_votes', {})
                for banner_key, vote_info in banner_votes.items():
                    voting_summary_prompt += f"  {banner_key}: {vote_info.get('vote', 'UNKNOWN')} - {vote_info.get('reasoning', 'No reasoning')}\n"
                
                if 'error' in vote_data:
                    voting_summary_prompt += f"  ERROR: {vote_data['error']}\n"
                
                voting_summary_prompt += "\n"
            
            voting_summary_prompt += f"""
            VOTING RULES:
            - Simple majority rule: 3+ RECOMMEND votes = survive, 2 or fewer = eliminate
            - Banner with fewest RECOMMEND votes gets eliminated
            - Provide final elimination decision and rationale
            
            Process all judge votes and determine final elimination using simple majority rule.
            Output your decision in the required JSON format.
            """
            
            # Create VotingCoordinator team
            coordinator_config = {
                "team": "VotingCoordination",
                "return": "FINISH",
                "prompt": "Process multi-judge votes and determine final elimination using simple majority rule.",
                "additional_prompt": "Apply majority voting logic and provide final decision with detailed reasoning.",
                "VotingCoordinator": prompt_template["team"]["VotingCoordinator"]
            }
            
            react_generator = agent_team.ReactAgent(
                intermediate_output_desc=prompt_template["intermediate_output_desc"], 
                config=self.config, 
                llm=self.llm
            )
            
            coordinator_team = agent_team.buildTeam(
                coordinator_config, 
                react_generator, 
                prompt_template["intermediate_output_desc"], 
                prompt_template["int_out_format"]
            )
            
            # Execute VotingCoordinator
            voting_result = coordinator_team(
                state={
                    "history": {
                        "VotingCoordination Supervisor": [HumanMessage(content=voting_summary_prompt)], 
                        "all": [HumanMessage(content=voting_summary_prompt)]
                    }, 
                    "intermediate_output": {},
                    "recent_files": recent_files,
                    "background": AIMessage(content="Multi-judge voting result processing and final elimination decision"),
                    "judge_votes": all_judge_votes  # Pass judge votes for reference
                },
                config={"recursion_limit": int(self.config["LLM"]["max_attempts"])}
            )
            
            # Extract final voting result
            if isinstance(voting_result, list):
                final_result = voting_result[0]["intermediate_output"]
            else:
                final_result = voting_result["intermediate_output"]
            
            # Store judge evaluations for history
            self.judge_evaluations.append({
                "round_number": self.round_number,
                "judge_votes": all_judge_votes,
                "final_decision": final_result
            })
            
            print(f"🎯 Multi-judge voting completed. Final decision: {final_result.get('worst_banner', 'No elimination')}")
            
            return final_result
            
        except Exception as e:
            print(f"❌ Error processing multi-judge votes: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Random elimination if voting fails
            return {
                "round_number": self.round_number,
                "active_banners": list(active_banner_data.keys()),
                "worst_banner": list(active_banner_data.keys())[-1] if len(active_banner_data) > 1 else None,
                "elimination_reason": f"Multi-judge voting failed: {e}. Random elimination applied.",
                "vote_summary": {},
                "judge_votes_detail": all_judge_votes
            }

    def _finalize_results(self, error_occurred=False):
        """
        Finalize and return the competition results
        """
        winner_id = self.active_banners[0] if self.active_banners else None
        
        print("\n" + "=" * 60)
        if error_occurred:
            print("⚠️ COMPETITION RESULTS (with errors)")
        else:
            print("🏆 COMPETITION RESULTS")
        print("=" * 60)
        
        if winner_id:
            print(f"🥇 WINNER: Banner #{winner_id}")
            print(f"🎨 Design Philosophy: {self.design_philosophies[winner_id]}")
        else:
            print("❌ No winner determined")
        
        print(f"\n📊 Elimination History:")
        if self.elimination_history:
            for elimination in self.elimination_history:
                print(f"   Round {elimination['round']}: Banner #{elimination['eliminated']} eliminated")
                print(f"   Reason: {elimination['reason']}")
        else:
            print("   No eliminations occurred")
        
        print(f"\n📈 Banner Results Summary:")
        for team_id in range(1, 6):
            if team_id in self.banner_results:
                if "error" in self.banner_results[team_id]:
                    print(f"   Banner #{team_id}: ❌ Error - {self.banner_results[team_id]['error']}")
                else:
                    print(f"   Banner #{team_id}: ✅ Success")
            else:
                print(f"   Banner #{team_id}: ❓ No result")
        
        return {
            "winner_banner_id": winner_id,
            "winner_banner_data": self.banner_results.get(winner_id, {}) if winner_id else {},
            "elimination_history": self.elimination_history,
            "judge_evaluations": self.judge_evaluations,
            "all_banner_results": self.banner_results,
            "final_round": self.round_number,
            "error_occurred": error_occurred
        }


def run_competitive_banner_generation(config, item_description, llm=None, logo_path=None, use_dynamic_styles=False):
    """
    Main function to run the competitive banner generation workflow
    
    Args:
        config: Configuration object (num_teams is read from config.ini)
        item_description: Description of the item to create banner for
        llm: Language model instance (optional)
        logo_path: Path to logo file for dynamic style generation
        use_dynamic_styles: Whether to use AI-generated styles instead of predefined ones
    
    Returns:
        Competition results dictionary
    """
    workflow = CompetitiveWorkflow(config, llm, logo_path=logo_path, use_dynamic_styles=use_dynamic_styles)
    return workflow.execute_competitive_generation(item_description)

# Main execution for testing
if __name__ == "__main__":
    import configparser
    from pathlib import Path

    # Load configuration
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.parent / "config" / "config_llm.ini"
    config.read(config_path)
    
    # 🎯 CONFIGURATION: num_teams is now controlled in config/config_llm.ini
    # You can modify [LLM] num_teams = 3 in config_llm.ini to change team count (1-5)
    try:
        NUM_TEAMS = int(config.get("LLM", "num_teams", fallback=5))
    except:
        NUM_TEAMS = 5
    
    print(f"🧪 Testing Competitive Banner Generation with {NUM_TEAMS} teams")
    print(f"📝 Team count is configured in config/config_llm.ini [LLM] num_teams = {NUM_TEAMS}")
    print("=" * 60)
    
    # Run the competitive workflow
    start_time = time.time()
    
    results = run_competitive_banner_generation(
        config=config,
        item_description="Sony digital camera with advanced photography features"
    )
    
    end_time = time.time()
    total_workflow_time = end_time - start_time
    
    # Display final results
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS")
    print("=" * 60)
    
    if results["winner_banner_id"]:
        print(f"🥇 Winner: Banner #{results['winner_banner_id']}")
        workflow = CompetitiveWorkflow(config)  # For accessing design_philosophies
        print(f"🎨 Design Style: {workflow.design_philosophies[results['winner_banner_id']]}")
    else:
        print("❌ No winner determined")
    
    print(f"\n📊 Competition Summary:")
    print(f"   Total Rounds: {results['final_round']}")
    print(f"   Banners Generated: {NUM_TEAMS} (CONCURRENT)")
    print(f"   Eliminations: {len(results['elimination_history'])}")
    print(f"   Total Workflow Time: {total_workflow_time:.2f}s")
    
    print(f"\n🗂️ Elimination History:")
    for elimination in results["elimination_history"]:
        print(f"   Round {elimination['round']}: Banner #{elimination['eliminated']} eliminated")
        print(f"      Reason: {elimination['reason'][:100]}...")
    
    # Save results
    output_dir = Path("outputs/competitive_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "competition_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"📄 Detailed results: {results_file}")
    print("\n🎉 Competition workflow completed!") 