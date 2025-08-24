import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

# Configuration for batch execution
MAX_THREADS = 4  # Maximum number of concurrent threads

# Resolve repository root dynamically (repo root is one level above this script)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Input and output directories relative to repo root (portable across environments)
QUESTION_FILE_DIR = REPO_ROOT / 'examples' / 'Problems' / 'wolftasks'
WTOOL_OUTPUT_TO_DIR = REPO_ROOT / 'output' / 'wolftasks_output_gmpro'
WOTOOL_OUTPUT_TO_DIR = REPO_ROOT / 'output' / 'wolftasks_nowolf_output_gmpro'
# list all .md file in QUESTION_FILE_DIR.
input_md_files = list(Path(QUESTION_FILE_DIR).glob("*.md"))

input_md_files = [file for file in input_md_files]

wtool_output_md_files = [
    Path(WTOOL_OUTPUT_TO_DIR) / f"A{file.stem[1:]}.md" for file in input_md_files
]
wotool_output_md_files = [
    Path(WOTOOL_OUTPUT_TO_DIR) / f"A{file.stem[1:]}.md" for file in input_md_files
]

ARGS_LIST = [
    *[{
        "manager_model": "openrouter/google/gemini-2.5-pro",
        "image_tool_model": "openrouter/google/gemini-2.5-pro",
        "review_tool_model": "openrouter/google/gemini-2.5-pro",
        "summarize_tool_model": "openrouter/google/gemini-2.5-pro",
        "manager_type": "CodeAgent",
        "tools_list": ["wolfram_alpha_query"],
        "input_markdown_file": f"{input_file}",
        "output_file_position": f"{output_file}",
    } for (input_file, output_file) in zip(input_md_files, wtool_output_md_files)],
    *[{
        "manager_model": "openrouter/google/gemini-2.5-pro",
        "image_tool_model": "openrouter/google/gemini-2.5-pro",
        "review_tool_model": "openrouter/google/gemini-2.5-pro",
        "summarize_tool_model": "openrouter/google/gemini-2.5-pro",
        "manager_type": "CodeAgent",
        "tools_list": [],
        "input_markdown_file": f"{input_file}",
        "output_file_position": f"{output_file}",
    } for (input_file, output_file) in zip(input_md_files, wotool_output_md_files)],
]

def build_command(args_dict):
    """Build the command to run run.py with the given arguments."""
    cmd = [
        sys.executable, "run.py",
        "--input-markdown-file", args_dict["input_markdown_file"],
        "--manager-model", args_dict["manager_model"],
        "--image-tool-model", args_dict["image_tool_model"],
        "--review-tool-model", args_dict["review_tool_model"],
        "--summarize-tool-model", args_dict["summarize_tool_model"],
    ]
    
    # Add optional arguments if present
    if "managed_agents_list_model" in args_dict:
        cmd.extend(["--managed-agents-list-model", args_dict["managed_agents_list_model"]])
    if 'manager_type' in args_dict:
        cmd.extend(["--manager-type", args_dict["manager_type"]])
    if "managed_agents_list" in args_dict:
        cmd.extend(["--managed-agents-list", *args_dict["managed_agents_list"]])
    
    # Add tools list
    cmd.append("--tools-list")
    cmd.extend(args_dict["tools_list"])
    
    return cmd

def run_single_task(args_dict):
    """Run a single task with the given arguments."""
    try:
        cmd = build_command(args_dict)
        print(f"Starting task: {args_dict['manager_model']} -> {args_dict['output_file_position']}")
        
        # Resolve absolute output path; if relative, place it under repo root
        output_path = Path(args_dict["output_file_position"]) 
        output_abs_path = output_path if output_path.is_absolute() else (REPO_ROOT / output_path)
        output_abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Build shell command with output redirection to absolute path
        quoted_cmd = ' '.join(f'"{arg}"' if (' ' in arg or '(' in arg or ')' in arg) else arg for arg in cmd)
        shell_cmd = f"{quoted_cmd} > \"{str(output_abs_path)}\" 2>&1"
        
        # Execute the command
        result = subprocess.run(
            shell_cmd,
            shell=True,
            cwd=str(REPO_ROOT)
        )
        
        if result.returncode == 0:
            print(f"✓ Task completed: {args_dict['manager_model']}")
        else:
            print(f"✗ Task failed: {args_dict['manager_model']} (code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Exception in task {args_dict['manager_model']}: {str(e)}")
        return False

def main():
    """Main function to run all tasks with multithreading."""
    print(f"🚀 Starting batch execution with {MAX_THREADS} concurrent threads")
    print(f"📋 Total tasks: {len(ARGS_LIST)}")
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(run_single_task, args_dict): args_dict 
            for args_dict in ARGS_LIST
        }
        
        # Process completed tasks
        completed = 0
        successful = 0
        
        for future in as_completed(future_to_args):
            args_dict = future_to_args[future]
            completed += 1
            
            try:
                success = future.result()
                if success:
                    successful += 1
                print(f"📊 Progress: {completed}/{len(ARGS_LIST)} tasks completed")
            except Exception as e:
                print(f"❌ Task failed with exception: {str(e)}")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🏁 Execution Summary:")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ Successful: {successful}/{len(ARGS_LIST)}")
    print(f"❌ Failed: {len(ARGS_LIST) - successful}")
    print(f"⚡ Average time per task: {total_time/len(ARGS_LIST):.2f} seconds")

if __name__ == "__main__":
    main()