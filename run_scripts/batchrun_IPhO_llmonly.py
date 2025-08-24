import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

# Configuration for batch execution
MAX_THREADS = 3  # Maximum number of concurrent threads
NUM_PER_QUESTION = 5 # number of runs per problem

ARGS_LIST = [
    *[{
        "model": "google/gemini-2.5-pro",
        "input_markdown_file": "examples/Problems/IPhO25/theory1/theory1.md",
        "image_base_dir": "examples/Problems/IPhO25/theory1/",
        "output_file_position": f"output/llmOnly_IPhO2025/theory1/theory1_{idx}.md",
        "log_file_position": f"output/llmOnly_IPhO2025/theory1/theory1_{idx}.log",
    } for idx in range(NUM_PER_QUESTION)],
    *[{
        "model": "google/gemini-2.5-pro",
        "input_markdown_file": "examples/Problems/IPhO25/theory2/theory2.md",
        "image_base_dir": "examples/Problems/IPhO25/theory2/",
        "output_file_position": f"output/llmOnly_IPhO2025/theory2/theory2_{idx}.md",
        "log_file_position": f"output/llmOnly_IPhO2025/theory2/theory2_{idx}.log",
    } for idx in range(NUM_PER_QUESTION)],
    *[{
        "model": "google/gemini-2.5-pro",
        "input_markdown_file": "examples/Problems/IPhO25/theory3/theory3.md",
        "image_base_dir": "examples/Problems/IPhO25/theory3/",
        "output_file_position": f"output/llmOnly_IPhO2025/theory3/theory3_{idx}.md",
        "log_file_position": f"output/llmOnly_IPhO2025/theory3/theory3_{idx}.log",
    } for idx in range(NUM_PER_QUESTION)],
]

def build_command(args_dict):
    """Build the command to run run.py with the given arguments."""
    cmd = [
        sys.executable, "-m", "run_scripts.run_llmonly",
        "--markdown_file", args_dict["input_markdown_file"],
        "--image_base_dir", args_dict["image_base_dir"],
        "--output_markdown", args_dict["output_file_position"],
        "--model", args_dict["model"],
    ]
    
    
    
    
    return cmd

def run_single_task(args_dict):
    """Run a single task with the given arguments."""
    try:
        cmd = build_command(args_dict)
        print(f"Starting task: {args_dict['model']} -> {args_dict['log_file_position']}")
        
        # Get the base directory (parent of run_scripts)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        
        # Create output directory if it doesn't exist (relative to base directory)
        output_dir = Path(base_dir) / Path(args_dict["log_file_position"]).parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build shell command with output redirection
        quoted_cmd = ' '.join(f'"{arg}"' if ' ' in arg or '(' in arg or ')' in arg else arg for arg in cmd)
        shell_cmd = f"{quoted_cmd} > \"{args_dict['log_file_position']}\" 2>&1"
        
        # Execute the command
        result = subprocess.run(
            shell_cmd,
            shell=True,
            cwd=base_dir
        )
        
        if result.returncode == 0:
            print(f"✓ Task completed: {args_dict['model']}")
        else:
            print(f"✗ Task failed: {args_dict['model']} (code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Exception in task {args_dict['model']}: {str(e)}")
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