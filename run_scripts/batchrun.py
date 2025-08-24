import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

# Configuration for batch execution
MAX_THREADS = 1  # Maximum number of concurrent threads

ARGS_LIST = [
    *[{
        "manager_model": "openrouter/google/gemini-2.5-flash",
        "image_tool_model": "openrouter/google/gemini-2.5-flash",
        "review_tool_model": "openrouter/google/gemini-2.5-flash",
        "summarize_tool_model": "openrouter/google/gemini-2.5-flash",
        "manager_type": "CodeAgent",
        "tools_list": ["wolfram_alpha_query", "ask_image_expert", "ask_review_expert"],
        "input_markdown_file": "examples/Problems/example/example1problem.md",
        "output_file_position": f"output/exampleProblem1/example1problem_gemini-2.5-flash__ToolAgent_wolfimgreview_{idx}.md",
    } for idx in range(1,2)],
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
        
        # Resolve repository root and ensure absolute output path
        repo_root = Path(__file__).resolve().parent.parent
        output_abs_path = (repo_root / args_dict["output_file_position"]).resolve()
        output_abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build shell command with output redirection to absolute path
        quoted_cmd = ' '.join(f'"{arg}"' if (' ' in arg or '(' in arg or ')' in arg) else arg for arg in cmd)
        shell_cmd = f"{quoted_cmd} > \"{str(output_abs_path)}\" 2>&1"
        
        # Execute the command
        result = subprocess.run(
            shell_cmd,
            shell=True,
            cwd=str(repo_root)
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