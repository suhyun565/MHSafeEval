"""
Patient Profile Loader - Project Structure Aware
For the structure:
    /home/station_06/Desktop/EmoAgent/
    ├── config/CCD/
    └── our_code_different/  ← code runs here
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class PatientProfileLoader:
    """Load and format patient profiles from JSON files"""
    
    def __init__(self, config_dir: str = None):
        """
        Args:
            config_dir: Base directory containing disorder subdirectories
                       If None, auto-detects based on project structure
        """
        if config_dir is None:
            config_dir = self._detect_project_config_dir()
        
        # Convert to absolute path
        self.config_dir = Path(config_dir).expanduser().resolve()
        
        if not self.config_dir.exists():
            print(f"⚠️  Warning: Config directory does not exist: {self.config_dir}")
            print(f"   Will create simple fallback profiles")
        else:
            print(f"✅ Config directory found: {self.config_dir}")
    
    def _detect_project_config_dir(self) -> str:
        """
        Auto-detect config directory for this project structure:
        /home/station_06/Desktop/EmoAgent/
        ├── config/CCD/  ← target
        └── our_code_different/  ← running from here
        """
        # Get current script directory
        current_dir = Path.cwd()
        
        # Strategy 1: Check relative to current working directory
        # If running from our_code_different/, go up one level
        relative_path = current_dir.parent / "config" / "CCD"
        if relative_path.exists():
            print(f"🔍 Auto-detected (parent): {relative_path}")
            return str(relative_path)
        
        # Strategy 2: Check one level up from current directory
        up_one = Path("../config/CCD").resolve()
        if up_one.exists():
            print(f"🔍 Auto-detected (..): {up_one}")
            return str(up_one)
        
        # Strategy 3: Check if we're in a subdirectory like our_code_different
        if "our_code_different" in str(current_dir):
            # Go up to parent, then to config/CCD
            project_root = current_dir.parent
            config_path = project_root / "config" / "CCD"
            if config_path.exists():
                print(f"🔍 Auto-detected (project root): {config_path}")
                return str(config_path)
        
        # Strategy 4: Absolute path (known location)
        absolute_path = Path("/home/station_06/Desktop/EmoAgent/config/CCD")
        if absolute_path.exists():
            print(f"🔍 Auto-detected (absolute): {absolute_path}")
            return str(absolute_path)
        
        # Strategy 5: Check common relative paths
        for rel_path in ["config/CCD", "../config/CCD", "../../config/CCD"]:
            test_path = Path(rel_path).resolve()
            if test_path.exists():
                print(f"🔍 Auto-detected: {test_path}")
                return str(test_path)
        
        # Default fallback (will create simple profiles)
        print(f"⚠️  Could not auto-detect config directory")
        print(f"   CWD: {current_dir}")
        print(f"   Using: ../config/CCD (may not exist)")
        return "../config/CCD"
    
    def load_patient_profile(
        self, 
        disorder_type: str, 
        patient_id: int
    ) -> Dict:
        """
        Load a single patient profile from JSON file
        
        Args:
            disorder_type: The disorder subdirectory (e.g., 'delusion', 'depression')
            patient_id: Patient number (1-indexed)
            
        Returns:
            Dictionary containing formatted patient profile
        """
        # Construct file path
        file_path = self.config_dir / disorder_type / f"patient{patient_id}.json"
        
        if not file_path.exists():
            # Don't print warning for every missing file (clutters output)
            # Only print for first few
            if patient_id <= 3:
                print(f"⚠️  Patient file not found: {file_path}")
                print(f"   Falling back to simple profile")
            return self._create_simple_profile(disorder_type, patient_id)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Format the profile for use in prompts
            formatted_profile = self._format_profile(data, disorder_type, patient_id)
            return formatted_profile
            
        except Exception as e:
            print(f"❌ Error loading profile from {file_path}: {e}")
            print(f"   Falling back to simple profile")
            return self._create_simple_profile(disorder_type, patient_id)
    
    def load_all_patients(
        self, 
        disorder_type: str, 
        num_patients: int
    ) -> List[Dict]:
        """
        Load multiple patient profiles
        
        Args:
            disorder_type: The disorder subdirectory
            num_patients: Number of patients to load
            
        Returns:
            List of formatted patient profiles
        """
        profiles = []
        loaded_count = 0
        fallback_count = 0
        
        for i in range(1, num_patients + 1):
            profile = self.load_patient_profile(disorder_type, i)
            profiles.append(profile)
            
            # Track loaded vs fallback
            if len(profile.get('life_history', '')) > 100:
                loaded_count += 1
            else:
                fallback_count += 1
        
        # Summary
        print(f"\n📋 Loaded {len(profiles)} patient profiles:")
        if loaded_count > 0:
            print(f"   ✅ {loaded_count} from JSON files")
        if fallback_count > 0:
            print(f"   ⚠️  {fallback_count} simple fallbacks")
        
        return profiles
    
    def _format_profile(
        self, 
        data: Dict, 
        disorder_type: str, 
        patient_id: int
    ) -> Dict:
        """
        Format raw JSON data into a structured profile
        
        Args:
            data: Raw data from JSON file
            disorder_type: Disorder type for this patient
            patient_id: Patient ID number
            
        Returns:
            Formatted profile dictionary
        """
        # Extract core information
        life_history = data.get("life_history", "No life history provided")
        
        # Format core beliefs
        core_beliefs = data.get("core_beliefs", [])
        core_belief_descriptions = data.get("core_belief_description", [])
        
        # Format intermediate beliefs
        intermediate_beliefs = data.get("intermediate_beliefs", [])
        delusion_beliefs = data.get("intermediate_beliefs_during_delusion", [])
        
        # Format coping strategies
        coping_strategies = data.get("coping_strategies", [])
        
        # Format cognitive models (if available)
        cognitive_models = data.get("cognitive_models", [])
        
        # Create formatted profile
        formatted = {
            "patient_id": patient_id,
            "name": f"Patient {patient_id}",
            "disorder": disorder_type,
            "age": 30 + patient_id,  # Default age range
            
            # Detailed information
            "life_history": life_history,
            "core_beliefs": core_beliefs,
            "core_belief_descriptions": core_belief_descriptions,
            "intermediate_beliefs": intermediate_beliefs,
            "intermediate_beliefs_during_delusion": delusion_beliefs,
            "coping_strategies": coping_strategies,
            "cognitive_models": cognitive_models,
            
            # Legacy fields for backward compatibility
            "background": life_history[:200] + "..." if len(life_history) > 200 else life_history,
            "current_state": self._infer_current_state(data),
        }
        
        return formatted
    
    def _infer_current_state(self, data: Dict) -> str:
        """Infer current emotional state from profile data"""
        beliefs = data.get("intermediate_beliefs_during_delusion", [])
        if beliefs:
            return "Currently experiencing distress; " + beliefs[0][:100]
        return "Seeking help for ongoing challenges"
    
    def _create_simple_profile(
        self, 
        disorder_type: str, 
        patient_id: int
    ) -> Dict:
        """
        Create a simple fallback profile when JSON is not available
        
        Args:
            disorder_type: Disorder type
            patient_id: Patient ID
            
        Returns:
            Simple profile dictionary
        """
        return {
            "patient_id": patient_id,
            "name": f"Patient {patient_id}",
            "age": 30 + patient_id,
            "disorder": disorder_type,
            "background": f"Patient {patient_id} with {disorder_type}",
            "current_state": "Seeking help",
            
            # Empty detailed fields
            "life_history": f"Patient {patient_id} seeking help for {disorder_type}",
            "core_beliefs": [],
            "core_belief_descriptions": [],
            "intermediate_beliefs": [],
            "intermediate_beliefs_during_delusion": [],
            "coping_strategies": [],
            "cognitive_models": [],
        }
    
    def format_profile_for_prompt(self, profile: Dict) -> str:
        """
        Format a profile dictionary into a readable string for prompts
        
        Args:
            profile: Patient profile dictionary
            
        Returns:
            Formatted string for use in prompts
        """
        lines = []
        
        # Basic info
        lines.append(f"Name: {profile['name']}")
        lines.append(f"Age: {profile['age']}")
        lines.append(f"Disorder: {profile['disorder']}")
        lines.append("")
        
        # Life history
        if profile.get("life_history") and len(profile["life_history"]) > 50:
            lines.append("Life History:")
            lines.append(profile["life_history"])
            lines.append("")
        
        # Core beliefs
        if profile.get("core_beliefs"):
            lines.append("Core Beliefs:")
            for belief in profile["core_beliefs"]:
                lines.append(f"  - {belief}")
            lines.append("")
        
        # Core belief descriptions
        if profile.get("core_belief_descriptions"):
            lines.append("Core Belief Descriptions:")
            for desc in profile["core_belief_descriptions"]:
                lines.append(f"  - {desc}")
            lines.append("")
        
        # Intermediate beliefs
        if profile.get("intermediate_beliefs"):
            lines.append("Intermediate Beliefs:")
            for belief in profile["intermediate_beliefs"]:
                lines.append(f"  - {belief}")
            lines.append("")
        
        # Beliefs during distress
        if profile.get("intermediate_beliefs_during_delusion"):
            lines.append("Beliefs During Distress:")
            for belief in profile["intermediate_beliefs_during_delusion"]:
                lines.append(f"  - {belief}")
            lines.append("")
        
        # Coping strategies
        if profile.get("coping_strategies"):
            lines.append("Coping Strategies:")
            for strategy in profile["coping_strategies"]:
                lines.append(f"  - {strategy}")
            lines.append("")
        
        # Cognitive models (examples)
        if profile.get("cognitive_models"):
            lines.append("Example Cognitive Patterns:")
            for i, model in enumerate(profile["cognitive_models"][:2], 1):  # Max 2 examples
                lines.append(f"  Example {i}:")
                lines.append(f"    Situation: {model.get('situation', 'N/A')}")
                lines.append(f"    Thoughts: {model.get('automatic_thoughts', 'N/A')}")
                lines.append(f"    Emotion: {model.get('emotion', 'N/A')}")
                lines.append(f"    Behavior: {model.get('behavior', 'N/A')}")
                lines.append("")
        
        # Current state
        lines.append(f"Current State: {profile.get('current_state', 'Seeking help')}")
        
        return "\n".join(lines)


# Convenience function for quick loading
def load_patient_profiles(
    disorder_type: str, 
    num_patients: int, 
    config_dir: str = None
) -> List[Dict]:
    """
    Quick function to load patient profiles
    
    Args:
        disorder_type: Disorder subdirectory name
        num_patients: Number of patients to load
        config_dir: Base config directory (auto-detects if None)
        
    Returns:
        List of patient profile dictionaries
    """
    loader = PatientProfileLoader(config_dir)
    return loader.load_all_patients(disorder_type, num_patients)


if __name__ == "__main__":
    # Test the loader
    print("="*80)
    print("TESTING PATIENT PROFILE LOADER")
    print("="*80)
    print()
    
    print("📍 Current Working Directory:")
    print(f"   {Path.cwd()}")
    print()
    
    print("📁 Testing auto-detection:")
    print("-"*80)
    
    # Test with auto-detection
    loader = PatientProfileLoader()
    
    print()
    print("📋 Testing profile loading:")
    print("-"*80)
    
    # Try loading a profile
    profile = loader.load_patient_profile("depression", 1)
    
    if len(profile.get('life_history', '')) > 100:
        print("✅ Successfully loaded detailed profile!")
        print(f"   Life history: {len(profile['life_history'])} characters")
        print(f"   Core beliefs: {len(profile.get('core_beliefs', []))}")
        print(f"   Coping strategies: {len(profile.get('coping_strategies', []))}")
    else:
        print("⚠️  Loaded simple fallback profile")
        print("   (JSON file not found or empty)")
    
    print()
    print("="*80)