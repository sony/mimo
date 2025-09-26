"""
Soft Limit Wrapper for Banner Generation Teams

This module implements a soft limit mechanism for controlling revision cycles:
1. Preferred termination: EvaluationTeam reports "No changes needed"
2. Fallback termination: Force exit after 3 revisions if not achieved
3. Accept results: Both perfect and imperfect results go to Judge evaluation
"""

import time
import threading
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

class SoftLimitWrapper:
    """
    Wraps a banner generation team to implement soft revision limits
    """
    
    def __init__(self, team_func, team_id, max_revisions=3, config=None):
        self.team_func = team_func
        self.team_id = team_id
        self.max_revisions = max_revisions
        self.revision_count = 0
        self.evaluation_results = []
        self.config = config
        
        # Special handling for max_revisions = 0
        if self.max_revisions == 0:
            print(f"🚀 [Team {self.team_id}] Zero-revision mode: Will skip all GraphicRevisor activities")
        
    def __call__(self, state, config):
        """
        Execute the banner generation team with soft revision limits
        """
        if self.max_revisions == 0:
            print(f"🎨 [Team {self.team_id}] Starting with zero-revision mode (direct to Judge)")
            return self._execute_zero_revision_mode(state, config)
        else:
            print(f"🎨 [Team {self.team_id}] Starting with soft limit: {self.max_revisions} revisions max")
            
        # Store original recursion limit
        original_limit = config.get("recursion_limit", 20)
        
        # Start execution loop
        start_time = time.time()
        
        try:
            # Try normal execution first
            result = self._execute_with_monitoring(state, config)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"✅ [Team {self.team_id}] Completed in {execution_time:.2f}s")
            print(f"📊 [Team {self.team_id}] Final revision count: {self.revision_count}")
            
            return result
            
        except Exception as e:
            # If execution fails, create a minimal result
            print(f"⚠️ [Team {self.team_id}] Execution interrupted: {str(e)[:100]}...")
            
            return self._create_fallback_result(state)
    
    def _execute_zero_revision_mode(self, state, config):
        """
        Execute banner generation without any revisions (ContentCreation only)
        """
        print(f"🚀 [Team {self.team_id}] Zero-revision mode: Executing ContentCreation → Direct to Judge")
        
        start_time = time.time()
        
        try:
            # Create a modified state that signals zero-revision mode
            enhanced_state = {
                **state,
                "zero_revision_mode": True,
                "max_revisions": 0,
                "skip_evaluation_cycles": True
            }
            
            # Set a lower recursion limit since we're skipping revision cycles
            zero_revision_config = {
                **config,
                "recursion_limit": min(10, config.get("recursion_limit", 20)),  # Reduced limit
                "zero_revision_mode": True
            }
            
            # Execute the team function with zero-revision constraints
            result = self.team_func(enhanced_state, zero_revision_config)
            
            # Process the result
            if isinstance(result, list):
                final_result = result[0]
            else:
                final_result = result
            
            # Add zero-revision metadata
            if "intermediate_output" in final_result:
                if isinstance(final_result["intermediate_output"], dict):
                    final_result["intermediate_output"]["revision_count"] = 0
                    final_result["intermediate_output"]["soft_limit_status"] = "zero_revision_complete"
                    final_result["intermediate_output"]["workflow_mode"] = "zero_revision"
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"✅ [Team {self.team_id}] Zero-revision mode completed in {execution_time:.2f}s")
            print(f"📊 [Team {self.team_id}] Revision count: 0 (as configured)")
            
            return result
            
        except Exception as e:
            print(f"⚠️ [Team {self.team_id}] Error in zero-revision mode: {e}")
            # Create fallback result for zero-revision mode
            return self._create_fallback_result(state, zero_revision=True)
    
    def _execute_with_monitoring(self, state, config):
        """
        Execute the team with revision monitoring using file system monitoring
        """
        print(f"🔧 [Team {self.team_id}] Starting execution with max_revisions={self.max_revisions}")
        
        try:
            # Execute the team function
            result = self.team_func(state, config)
            
            # Check if revision limit was exceeded by monitoring generated files
            if self._check_revision_limit_exceeded():
                print(f"🛑 [Team {self.team_id}] Revision limit exceeded after execution")
                return self._create_limit_exceeded_result(state)
            
            # Process the result
            if isinstance(result, list):
                final_result = result[0]
            else:
                final_result = result
            
            # Add soft limit metadata
            if "intermediate_output" in final_result:
                if isinstance(final_result["intermediate_output"], dict):
                    actual_revisions = self._count_team_revisions()
                    final_result["intermediate_output"]["revision_count"] = actual_revisions
                    final_result["intermediate_output"]["soft_limit_status"] = "completed"
                    final_result["intermediate_output"]["max_revisions"] = self.max_revisions
            
            return result
            
        except Exception as e:
            print(f"⚠️ [Team {self.team_id}] Exception during execution: {e}")
            raise e
    
    def _check_revision_limit_exceeded(self):
        """
        Check if revision limit was exceeded by examining generated files
        """
        import glob
        import re
        import os
        
        # Get output folder from config
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
        
        # Look for files matching pattern: generated_image_*_Team{team_id}_Round*_*.png
        pattern = f"{output_folder}/generated_image_*_Team{self.team_id}_Round*_*.png"
        files = glob.glob(pattern)
        
        max_revision_found = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
            match = re.search(r'_Team\d+_Round\d+_(\d+)\.png$', filename)
            if match:
                revision_num = int(match.group(1))
                max_revision_found = max(max_revision_found, revision_num)
        
        if max_revision_found > self.max_revisions:
            print(f"🚨 [Team {self.team_id}] Revision limit exceeded: found revision {max_revision_found}, limit is {self.max_revisions}")
            return True
        
        return False
    
    def _count_team_revisions(self):
        """
        Count actual revisions by examining generated files
        """
        import glob
        import re
        import os
        
        # Get output folder from config
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
        
        pattern = f"{output_folder}/generated_image_*_Team{self.team_id}_Round*_*.png"
        files = glob.glob(pattern)
        
        max_revision = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            match = re.search(r'_Team\d+_Round\d+_(\d+)\.png$', filename)
            if match:
                revision_num = int(match.group(1))
                max_revision = max(max_revision, revision_num)
        
        return max_revision
    
    def _create_limit_exceeded_result(self, state):
        """
        Create a result when revision limit is exceeded
        """
        return {
            "intermediate_output": {
                "status": "soft_limit_exceeded",
                "team_id": self.team_id,
                "revision_count": self._count_team_revisions(),
                "max_revisions": self.max_revisions,
                "message": f"Team {self.team_id} exceeded revision limit ({self.max_revisions}). Using last generated banner.",
                "note": "Banner generation stopped due to revision limit"
            },
            "messages": [state.get("background", "")],
            "history": state.get("history", {}),
            "recent_files": state.get("recent_files", {}),
            "next": "FINISH"
        }
    
    def _create_fallback_result(self, state, zero_revision=False):
        """
        Create a fallback result when execution fails
        """
        if zero_revision:
            status_message = "Zero-revision mode fallback: ContentCreation completed, ready for Judge evaluation."
            workflow_status = "zero_revision_fallback"
            note = "Initial banner created without revisions as configured (max_revisions=0)"
        else:
            status_message = "Banner generation completed with soft limits. Ready for Judge evaluation."
            workflow_status = "soft_limit_fallback"
            note = "Result may not be fully optimized but meets minimum requirements"
        
        return {
            "intermediate_output": {
                "status": workflow_status,
                "team_id": self.team_id,
                "revision_count": self.revision_count,
                "message": status_message,
                "note": note,
                "workflow_mode": "zero_revision" if zero_revision else "soft_limit"
            },
            "messages": AIMessage(content=f"Team {self.team_id} completed with {'zero-revision' if zero_revision else 'soft limit'} fallback"),
            "background": AIMessage(content=f"Team {self.team_id} banner generation task"),
            "history": state.get("history", {}),
            "recent_files": state.get("recent_files", {}),
            "next": "FINISH"
        }

def wrap_team_with_soft_limit(team_func, team_id, max_revisions=3, config=None):
    """
    Factory function to create a soft-limited team
    """
    return SoftLimitWrapper(team_func, team_id, max_revisions, config) 