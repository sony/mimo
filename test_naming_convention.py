#!/usr/bin/env python3
"""
Test new file naming convention with logo integration
Verify if file format meets user expectations:
generated_image_ImageResearcher_Team{TeamID}_Round{RoundNumber}_{RevisionNumber}.png

Test scenarios:
1. First round: 3 teams each generate 3 revisions using logo input, final files should be:
   - generated_image_ImageResearcher_Team1_Round1_3.png
   - generated_image_ImageResearcher_Team2_Round1_3.png  
   - generated_image_ImageResearcher_Team3_Round1_3.png

2. Second round: remaining 2 teams each generate 3 revisions, final files should be:
   - generated_image_ImageResearcher_Team1_Round2_3.png
   - generated_image_ImageResearcher_Team3_Round2_3.png

3. Third round: final winner revises again, final file should be:
   - generated_image_ImageResearcher_Team1_Round3_3.png
"""

import configparser
import sys
import os
import time
import glob
import argparse
from pathlib import Path

# Add path for module imports
sys.path.append('src/multiagent')
sys.path.append('src')

from multiagent.competitive_workflow import run_competitive_banner_generation


def find_logo_and_prompt(product_name="ethicai"):
    """
    Find logo and prompt files for the given product name
    Returns tuple of (logo_path, prompt_text)
    """
    # Try exact match first
    exact_logo_files = glob.glob(f"logos/*{product_name.lower()}*.png")
    
    # If exact match found, use it
    if exact_logo_files:
        logo_path = exact_logo_files[0]
        print(f"🎯 Found exact match for '{product_name}': {logo_path}")
    else:
        # If no exact match, list available logos and exit
        all_logos = glob.glob("logos/*.png")
        if all_logos:
            print(f"❌ No logo file found containing '{product_name}'")
            print("📁 Available logo files:")
            for logo in sorted(all_logos):
                basename = os.path.splitext(os.path.basename(logo))[0]
                # Extract logo name (remove number prefix if exists)
                logo_name = basename.split('_', 1)[-1] if '_' in basename else basename
                print(f"   {logo_name} (file: {os.path.basename(logo)})")
            print(f"\n💡 Usage: python {sys.argv[0]} --logo <logo name>")
            return None, None
        else:
            print(f"❌ No PNG files found in logos/ directory")
            return None, None
    
    # Find corresponding prompt file
    logo_basename = os.path.splitext(os.path.basename(logo_path))[0]
    
    # Look for corresponding prompt file - more flexible matching
    prompt_files = [
        f"logos/{logo_basename}_prompt.txt",
        f"logos/{product_name.lower()}_prompt.txt",
    ]
    
    prompt_text = None
    for prompt_file in prompt_files:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip().strip('"')
            print(f"📄 Using prompt file: {prompt_file}")
            break
    
    if prompt_text is None:
        # Generate default prompt based on logo name
        logo_name = logo_basename.split('_', 1)[-1] if '_' in logo_basename else logo_basename
        prompt_text = f"Create a professional banner for {logo_name}, showcasing high-quality design and brand identity"
        print(f"⚠️ No prompt file found, using default prompt: {prompt_text}")
    
    print(f"🖼️ Using logo: {logo_path}")
    print(f"📝 Using prompt: {prompt_text}")
    
    return logo_path, prompt_text


def test_naming_convention(logo_name=None, use_dynamic_styles=False, style_candidates=8):
    print("🧪 Using logo integration test for new file naming convention")
    if use_dynamic_styles:
        print("🎨 Enable dynamic style generation mode")
        print(f"📊 Style candidate count: {style_candidates} → Select best {num_teams if 'num_teams' in locals() else '3'}")
    print("=" * 60)
    
    # Use provided logo name or default
    if logo_name is None:
        logo_name = "ethicai"
    
    # Find logo and prompt files
    logo_path, prompt_text = find_logo_and_prompt(logo_name)
    
    if logo_path is None:
        return None
    
    # Load configuration
    config = configparser.ConfigParser()
    config_path = Path("config/config_llm.ini")
    config.read(config_path)
    
    # Set test parameters
    test_output_dir = f"{args.outputs_dir}/naming_test_{int(time.time())}"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy logo to output directory for easier access
    logo_filename = os.path.basename(logo_path)
    test_logo_path = os.path.join(test_output_dir, logo_filename)
    import shutil
    shutil.copy2(logo_path, test_logo_path)
    print(f"📋 Copied logo to test directory: {test_logo_path}")
    
    # Modify configuration to use test directory
    if not config.has_section('SETTING'):
        config.add_section('SETTING')
    config.set('SETTING', 'output_folder', test_output_dir)
    
    # read team count from config file, do not override
    try:
        num_teams = int(config.get('LLM', 'num_teams', fallback=3))
    except:
        num_teams = 3  # fallback value
    print(f"📊 Read team count from config file: {num_teams}")
    # do not override num_teams setting in config file
    config.set('LLM', 'max_revisions_per_team', '3')  # 1 revision per team for quick test
    
    # 🎯 MODIFICATION: Ensure ImageResearcher uses tool 12 (GeminiImageGenTool) instead of 13
    # Look for ImageResearcher agent configuration and update tools
    try:
        current_tools = config.get('LLM', 'image_researcher_tools', fallback='[12]')
        print(f"📊 Current ImageResearcher tools: {current_tools}")
        # Force use tool 12 (GeminiImageGenTool) instead of 13 (RecraftImageGenTool)
        config.set('LLM', 'image_researcher_tools', '[12]')
        print(f"🔧 Updated ImageResearcher to use tool 12 (GeminiImageGenTool)")
    except Exception as e:
        print(f"⚠️ Could not modify tool configuration: {e}")
    
    print(f"📁 Test output directory: {test_output_dir}")
    print(f"🎯 Number of teams: {num_teams}")
    print(f"🔧 Max revisions per team: 3") 
    print(f"🎨 Logo file: {logo_filename}")
    print(f"📝 Product description: {prompt_text}")
    if use_dynamic_styles:
        print(f"🎭 Dynamic styles: Enabled (candidate: {style_candidates} → select: {num_teams})")
    else:
        print(f"🎭 Dynamic styles: Use predefined styles")
    print()
    
    try:
        # Run competitive workflow with logo-enhanced prompt
        start_time = time.time()
        
        # Enhanced item description that includes logo reference and specific instructions for tool 12
        enhanced_description = f"{prompt_text}. IMPORTANT FOR IMAGE RESEARCHER: Use the logo image file '{logo_filename}' located in the output directory as input to the image editing tool (tool 12). You must provide the logo file path '{test_logo_path}' as input_filepath parameter to create a professional banner that incorporates and enhances the existing logo design. This is a PNG image file with transparent background, which makes it perfect for seamless integration into any banner design without background conflicts."
        
        # Store configuration for dynamic styles (avoid ConfigParser interpolation issues)
        config.set('SETTING', 'logo_file_path', test_logo_path)
        config.set('SETTING', 'logo_filename', logo_filename)
        config.set('SETTING', 'use_dynamic_styles', str(use_dynamic_styles))  # Dynamic styles开关
        config.set('SETTING', 'style_candidates', str(style_candidates))  # Dynamic styles候选数量
        # Note: enhanced_description is passed directly as parameter, no need to store in config
        
        # 🎨 NEW: Pass dynamic style parameters if enabled
        kwargs = {
            'config': config,
            'item_description': enhanced_description
        }
        
        if use_dynamic_styles:
            kwargs['logo_path'] = test_logo_path
            kwargs['use_dynamic_styles'] = True
        
        results = run_competitive_banner_generation(**kwargs)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n📊 Test completed! Total time: {total_time:.2f}s")
        print("=" * 60)
        
        # Verify file naming
        print("\n🔍 Verifying file naming convention...")
        verify_naming_convention(test_output_dir, results)
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_naming_convention(output_dir, results):
    """Verify that generated files follow expected naming convention"""
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory does not exist: {output_dir}")
        return
    
    # Get all generated image files
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    image_files.sort()
    
    print(f"📷 Total {len(image_files)} image files generated")
    
    if not image_files:
        print("❌ No image files found")
        return
    
    # Group by rounds
    rounds = {}
    teams = set()
    
    expected_pattern = r"generated_image_(\w+)_Team(\d+)_Round(\d+)_(\d+)\.png"
    import re
    
    print("\n📋 File list:")
    for filename in image_files:
        print(f"   {filename}")
        
        # Parse filename
        match = re.match(expected_pattern, filename)
        if match:
            agent_role, team_id, round_num, revision_num = match.groups()
            team_id = int(team_id)
            round_num = int(round_num)
            revision_num = int(revision_num)
            
            teams.add(team_id)
            
            if round_num not in rounds:
                rounds[round_num] = {}
            if team_id not in rounds[round_num]:
                rounds[round_num][team_id] = []
            
            rounds[round_num][team_id].append({
                'filename': filename,
                'agent_role': agent_role,
                'revision': revision_num
            })
        else:
            print(f"⚠️ Filename does not match expected pattern: {filename}")
    
    print(f"\n🎯 Teams found: {sorted(teams)}")
    print(f"🎯 Rounds found: {sorted(rounds.keys())}")
    
    # Verify files for each round
    print("\n🔍 Verification by round:")
    for round_num in sorted(rounds.keys()):
        print(f"\n📅 Round {round_num}:")
        round_data = rounds[round_num]
        
        for team_id in sorted(round_data.keys()):
            team_files = round_data[team_id]
            max_revision = max(f['revision'] for f in team_files)
            final_file = next(f for f in team_files if f['revision'] == max_revision)
            
            print(f"   Team {team_id}: {len(team_files)} files, final revision: {max_revision}")
            print(f"      Final file: {final_file['filename']}")
            
            # Verify naming format
            expected_final = f"generated_image_GraphicRevisor_Team{team_id}_Round{round_num}_{max_revision}.png"
            if final_file['filename'] == expected_final:
                print(f"      ✅ Naming format correct")
            else:
                print(f"      ❌ Naming format incorrect")
                print(f"         Expected: {expected_final}")
                print(f"         Actual: {final_file['filename']}")
    
    # Summarize verification results
    print(f"\n📊 Verification summary:")
    print(f"   Total files: {len(image_files)}")
    print(f"   Participating teams: {len(teams)}")
    print(f"   Competition rounds: {len(rounds)}")
    
    # Check for winner
    if results and 'winner_banner_id' in results and results['winner_banner_id']:
        winner_id = results['winner_banner_id']
        print(f"   🏆 Winner: Team {winner_id}")
        
        # Find winner's final image
        final_round = max(rounds.keys()) if rounds else 0
        if final_round in rounds and winner_id in rounds[final_round]:
            winner_files = rounds[final_round][winner_id]
            max_revision = max(f['revision'] for f in winner_files)
            winner_final_file = next(f for f in winner_files if f['revision'] == max_revision)
            print(f"   🎯 Winner's final file: {winner_final_file['filename']}")
            
            expected_winner_file = f"generated_image_GraphicRevisor_Team{winner_id}_Round{final_round}_{max_revision}.png"
            if winner_final_file['filename'] == expected_winner_file:
                print(f"   ✅ Winner file naming correct")
            else:
                print(f"   ❌ Winner file naming incorrect")
    
    # Keep files for inspection
    print(f"\n📁 Test files saved in: {output_dir}")
    print("   (Files will be retained for manual inspection)")


def parse_arguments():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test file naming convention and logo integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage examples:
  # Use predefined styles
  python test_naming_convention.py --logo ethicai
  
  # Use dynamic styles (default 8 candidates, select best 3)
  python test_naming_convention.py --logo wildcare --dynamic-styles
  
  # Customize style candidate count (generate 10 candidates, select best 5)
  python test_naming_convention.py --logo sparklekitch --dynamic-styles --style-candidates 10
  
Available logos (detected from logos/ directory):
  Run script without parameters to view full list
        '''
    )
    
    parser.add_argument(
        '--logo', 
        type=str, 
        default='ethicai',
        help='Specify the logo name to use (default: ethicai)'
    )
    
    parser.add_argument(
        '--list-logos',
        action='store_true',
        help='List all available logo files'
    )
    
    parser.add_argument(
        '--dynamic-styles',
        action='store_true',
        help='Use AI to dynamically generate design styles, instead of predefined 5 fixed styles'
    )
    
    parser.add_argument(
        '--style-candidates',
        type=int,
        default=8,
        help='Number of candidate styles generated in dynamic style mode (default: 8), will select best num_teams styles'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='experiment_results',
    )
    parser.add_argument(
        '--outputs_dir',
        type=str,
        default='outputs',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='past_cache',
    )
    
    return parser.parse_args()


def list_available_logos():
    """list all available logo files"""
    all_logos = glob.glob("logos/*.png")
    if not all_logos:
        print("❌ No PNG files found in logos/ directory")
        return
    
    print("📁 Available logo files:")
    print("=" * 40)
    for logo in sorted(all_logos):
        basename = os.path.splitext(os.path.basename(logo))[0]
        # Extract logo name (remove number prefix if exists)
        logo_name = basename.split('_', 1)[-1] if '_' in basename else basename
        
        # Check if corresponding prompt file exists
        prompt_file = f"logos/{basename}_prompt.txt"
        has_prompt = "✅" if os.path.exists(prompt_file) else "❌"
        
        print(f"   {logo_name:20} | file: {os.path.basename(logo):25} | prompt file: {has_prompt}")
    
    print(f"\n💡 Usage: python {sys.argv[0]} --logo <logo name>")


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.list_logos:
        list_available_logos()
        sys.exit(0)
    
    print("🎯 Start file naming convention test...")
    print("Expected format: generated_image_GraphicRevisor_Team{ID}_Round{R}_{Rev}.png")
    print(f"🎨 Using logo: {args.logo}")
    
    if args.dynamic_styles:
        print(f"🎭 Enable dynamic style generation: AI will analyze logo and prompt")
        print(f"   📊 Candidate pool: {args.style_candidates} styles → Smartly select best {num_teams if 'num_teams' in locals() else '3'}")
        print(f"   🎯 Selection strategy: AI evaluation + rule fallback")
    else:
        print("🎭 Use predefined styles: 5 fixed design styles")
    
    print()
    
    results = test_naming_convention(args.logo, args.dynamic_styles, args.style_candidates)
    
    if results:
        print("\n🎉 Test completed! Please check if the output file naming convention meets requirements.")
        sys.exit(0)  # 成功退出
    else:
        print("\n❌ Test failed, please check error messages.")
        sys.exit(1)  # 失败退出