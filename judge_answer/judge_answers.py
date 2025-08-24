import os
import re
import json
from typing import Optional, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from judge_answer.call_llm_utils import send_multimodal_message
NUM_DIGITS_JUDGED = os.environ.get("NUM_DIGITS_JUDGED", 4)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Load from .env file in the current directory
    print("dotenv loaded successfully from .env file")
    print(f"Currently using api key: {os.environ.get('OPENROUTER_API_KEY')[:20]}...")
except ImportError:
    # dotenv not available, continue without it
    print("dotenv not available, continuing without it")
    pass
except Exception:
    # .env file doesn't exist or other error, continue without raising exception
    print("Error loading .env file, continuing without it")
    pass

def extract_final_answer_from_model_response(content: str, max_chars: int = 10000) -> str:
    """
    Extract the final answer from a model response by taking the last max_chars characters
    and using LLM to extract the final answer.
    
    Args:
        content: The full model response content
        max_chars: Maximum number of characters to consider from the end
        
    Returns:
        The extracted final answer
    """
    # Take the last max_chars characters
    if len(content) > max_chars:
        content = content[-max_chars:]
    
    # Use LLM to extract the final answer
    prompt = f"""You are a helpful assistant that extracts the final mathematical answer from a physics problem solution.

Given the following solution text, extract ONLY the final mathematical answer. The answer should be:
1. A mathematical expression or formula
2. The final result that answers the original question
3. In LaTeX format if it contains mathematical notation

Solution text:
{content}

Please extract and return ONLY the final mathematical answer, use \[ and \] to wrap the answer:"""

    # You'll need to provide your API key here
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    try:
        result = send_multimodal_message(
            api_key=api_key,
            text=prompt,
            model="openai/gpt-4.1",
            # model="google/gemini-2.5-pro",
            return_token_consumption=False
        )
        return result.strip().replace('$', '').replace("\\[", "").replace("\\]", "").replace("...","")
    except Exception as e:
        print(f"Error extracting final answer: {e}")
        return ""


def judge_answer_similarity(model_answer: str, official_answer: str, api_key: str) -> dict:
    if "=" in model_answer:
        model_answer = model_answer.split("=")[1].strip().split("\n")[0].strip()
    if "=" in official_answer:
        official_answer = official_answer.split("=")[1].strip().split("\n")[0].strip()
    """
    Use LLM to judge whether the model answer is the same as the official answer.
    
    Args:
        model_answer: The extracted final answer from the model
        official_answer: The official answer
        api_key: OpenRouter API key
        
    Returns:
        Dictionary with judgment results
    """
    
    example1=r'''Standard answer: $$R\left(\frac{1}{\sqrt{k_2(1+k_1)-k_1}}-1\right)$$
  - Extracted answer: \[
h = R \left( \frac{1}{\sqrt{k_2 (1 + k_1) - k_1}} - 1 \right)
\]
    result: correct'''
    example2=r'''✓ Completed A41.md
  - Standard answer: R\left(\frac{1}{\sqrt{k_2(1+k_1)-k_1}}-1\right)
  - Extracted answer: 
h = R \left[ \frac{1}{\sqrt{k_2 (1 + k_1) - k_1}} - 1 \right]
...
  - Result: correct'''
    example_result=r'''```json
{
    "equivalent": true,
    "confidence": "high",
    "reasoning": "The model answer is correct."
}
```'''
    prompt = f"""You are an expert physics and mathematics evaluator. Your task is to determine if two mathematical expressions are equivalent.
    
Model Answer:
{model_answer}

Official Answer:
{official_answer}

Please evaluate if these answers are mathematically equivalent. Consider:
1. Mathematical equivalence (same numerical/analytical result). Note that the model/standard answer may only contains the expression but not the existing var. E.g. 'F=ma' and 'ma', this is considered as correct. Moreover,equivalent but different latex expressions are considered as correct.
2. Different but equivalent forms of the same expression
3. Minor notational differences (e.g., different variable names that represent the same thing)
4. Algebraic manipulations that lead to the same result
5. For inequalities, you may neglect difference in '='. E.g. 'a>0' and 'a\geq 0' are viewed as equivalent. Moreover, things like abs can be viewed as equivalent, for example, sqrt(a^2) and a can be viewed as equivalent.
6. Only JUDGE with the FIRST {NUM_DIGITS_JUDGED} effective digits (\pm 1 for the {NUM_DIGITS_JUDGED} digits), e.g. 0.123456789 and 0.123457 are equivalent when considering the first SIX digits.

For example,
{example1}

{example2}


Please think step by step and check carefully before your answer. Use <think></think> to indicate your thinking process, and <result></result> to indicate your final answer. In <result></result>, you should use either 'equivalent' or 'inequivalent' to indicate your final answer.

For example, 

<think>
I think the model answer is correct.
</think>
<result>equivalent</result>

For example,

<think>
I think the model answer is incorrect.
</think>
<result>inequivalent</result>

Response:"""

    try:
        result = send_multimodal_message(
            api_key=api_key,
            text=prompt,
            model="google/gemini-2.5-pro",
            # model="openai/gpt-4.1",
            return_token_consumption=False
        )
        
        import re
        import regex
        think_prompt = regex.search(r'<think>(.*?)</think>', result)
        if think_prompt:
            think_prompt = think_prompt.group(1).strip()
        else:
            think_prompt = ""
        # support multi-line json
        equiv_result = regex.search(r'<result>(.*?)</result>', result)
        if equiv_result:
            equiv_result = equiv_result.group(1).strip()
        else:
            equiv_result = ""
        if equiv_result == "equivalent":
            return {
                "equivalent": True,
                "confidence": "?",
                "reasoning": result
            }
        else:
            return {
                "equivalent": False,
                "confidence": "?",
                "reasoning": result
            }
        
        
        
            
    except Exception as e:
        return {
            "equivalent": "error",
            "confidence": "low",
            "reasoning": f"Error during evaluation: {str(e)}",
            "model_answer": model_answer,
            "official_answer": official_answer
        }



def process_single_problem(problem_info: dict, api_key: str, progress_lock: Lock) -> dict:
    """
    Process a single problem (thread-safe function).
    
    Args:
        problem_info: Dictionary containing problem information
        api_key: OpenRouter API key
        progress_lock: Lock for thread-safe progress printing
        
    Returns:
        Dictionary with problem results
    """
    model_file = problem_info['model_file']
    model_path = problem_info['model_path'] 
    official_path = problem_info['official_path']
    
    try:
        # Read model answer
        with open(model_path, 'r', encoding='utf-8') as f:
            model_content = f.read()
        
        # Extract final answer from model response
        model_final_answer = extract_final_answer_from_model_response(model_content)
        
        # Read official answer
        with open(official_path, 'r', encoding='utf-8') as f:
            official_answer = f.read().strip().replace('$$', '')
        
        if "=" in model_final_answer:
            model_final_answer = model_final_answer.split("=")[1].strip()
        if "=" in official_answer:
            official_answer = official_answer.split("=")[1].strip()
        
        
        # Judge similarity
        judgment = judge_answer_similarity(model_final_answer, official_answer, api_key)
        
        # Determine if correct or incorrect
        is_correct = judgment.get('equivalent', False) == True
        result_status = "correct" if is_correct else "incorrect"
        
        # Store details for this problem
        problem_name = model_file.replace('.md', '')
        result = {
            'problem_name': problem_name,
            'is_correct': is_correct,
            'details': {
                "standard_answer": official_answer,
                "llm_extracted_answer": model_final_answer,
                "llm_judged_result": result_status,
                "judgment_details": {
                    "confidence": judgment.get('confidence', 'unknown'),
                    "reasoning": judgment.get('reasoning', 'No reasoning provided')
                },
            }
        }
        
        # Thread-safe progress update
        with progress_lock:
            print(f"✓ Completed {model_file}")
            print(f"  - Standard answer: {official_answer}")
            print(f"  - Extracted answer: {model_final_answer}")
            print(f"  - LLM Judgment: {judgment}")
            print(f"  - LLM Result: {result_status}")
            print()
        
        return result
        
    except Exception as e:
        with progress_lock:
            print(f"✗ Error processing {model_file}: {str(e)}")
        
        return {
            'problem_name': model_file.replace('.md', ''),
            'is_correct': False,
            'details': {
                "standard_answer": "",
                "llm_extracted_answer": "",
                "llm_judged_result": "error",
                "judgment_details": {
                    "confidence": "low",
                    "reasoning": f"Error during processing: {str(e)}"
                }
            }
        }


def process_model_directory(model_dir: str, official_dir: str, api_key: str, output_file: str = "0_judged_result.json", max_workers: int = 4) -> dict:
    """
    Process all model answers that exist in both directories using multi-threading.
    
    Args:
        model_dir: Directory containing model answers
        official_dir: Directory containing official answers
        api_key: OpenRouter API key
        output_file: File to save results (default: "0_judged_result.json")
        max_workers: Maximum number of worker threads (default: 4)
        
    Returns:
        Dictionary with results for all processed files
    """
    if output_file is None:
        output_file = os.path.join(model_dir, "0_judged_result.json")
    else:
        output_file = os.path.join(model_dir, output_file)
    
    # Get files that exist in both directories
    model_files = set(f for f in os.listdir(model_dir) if f.endswith('.md'))
    official_files = set(f for f in os.listdir(official_dir) if f.endswith('.md'))
    common_files = model_files.intersection(official_files)
    
    print(f"Found {len(model_files)} model files, {len(official_files)} official files")
    print(f"Processing {len(common_files)} common problems with {max_workers} workers...")
    print(f"Starting multi-threaded processing...")
    print("=" * 60)
    
    # Prepare problem information for workers
    problem_infos = []
    for model_file in sorted(common_files):
        problem_infos.append({
            'model_file': model_file,
            'model_path': os.path.join(model_dir, model_file),
            'official_path': os.path.join(official_dir, model_file)
        })
    
    # Initialize counters and results
    details = {}
    judged_as_correct = 0
    judged_as_wrong = 0
    progress_lock = Lock()
    completed_count = 0
    
    start_time = time.time()
    
    # Process problems in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_problem = {
            executor.submit(process_single_problem, problem_info, api_key, progress_lock): problem_info
            for problem_info in problem_infos
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_problem):
            problem_info = future_to_problem[future]
            try:
                result = future.result()
                
                # Update counters
                if result['is_correct']:
                    judged_as_correct += 1
                else:
                    judged_as_wrong += 1
                
                # Store details
                details[result['problem_name']] = result['details']
                
                completed_count += 1
                
                # Progress update (thread-safe)
                with progress_lock:
                    elapsed_time = time.time() - start_time
                    progress_percent = (completed_count / len(problem_infos)) * 100
                    avg_time_per_problem = elapsed_time / completed_count
                    eta = avg_time_per_problem * (len(problem_infos) - completed_count)
                    
                    print(f"Progress: {completed_count}/{len(problem_infos)} ({progress_percent:.1f}%) | "
                          f"Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s")
                    print("-" * 40)
                
            except Exception as e:
                problem_name = problem_info['model_file'].replace('.md', '')
                print(f"Failed to process {problem_info['model_file']}: {e}")
                
                # Add error result
                details[problem_name] = {
                    "standard_answer": "",
                    "llm_extracted_answer": "",
                    "llm_judged_result": "error",
                    "judgment_details": {
                        "confidence": "low",
                        "reasoning": f"Processing failed: {str(e)}"
                    }
                }
                judged_as_wrong += 1
                completed_count += 1
    
    total_time = time.time() - start_time
    
    # Calculate accuracy rate
    total_problems = len(common_files)
    acc_rate = (judged_as_correct / total_problems) if total_problems > 0 else 0.0
    
    # Prepare final results
    final_results = {
        "number_of_problems": total_problems,
        "judged_as_correct": judged_as_correct,
        "judged_as_wrong": judged_as_wrong,
        "acc_rate": round(acc_rate, 4),
        "processing_time_seconds": round(total_time, 2),
        "avg_time_per_problem": round(total_time / total_problems, 2) if total_problems > 0 else 0,
        "details": details
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print(f"✅ Multi-threaded processing completed!")
    print(f"Results saved to {output_file}")
    print(f"Total time: {total_time:.2f}s | Average per problem: {total_time/total_problems:.2f}s")
    
    return final_results


def main():
    """Main function to run the answer judging process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge model answers against official answers using LLM")
    parser.add_argument("--model_dir", default='/mnt/e/agents/Olympiads/PS_phybench/mkd_model_answers/openrouter_google_gemini-2.5-pro_R02_wolf', help="Directory containing model answers")
    parser.add_argument("--official_dir", default='/mnt/e/agents/Olympiads/PS_phybench/mkd_model_answers/official_answer/', help="Directory containing official answers")
    parser.add_argument("--api_key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--output", default=None, help="Output file to save results")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads (default: 4)")
    
    args = parser.parse_args()
    
    if args.api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Please provide API key via --api_key argument or OPENROUTER_API_KEY environment variable")
        return
    
    # Process the directory with multi-threading
    results = process_model_directory(args.model_dir, args.official_dir, api_key, args.output, args.max_workers)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total problems processed: {results['number_of_problems']}")
    print(f"Judged as correct: {results['judged_as_correct']}")
    print(f"Judged as wrong: {results['judged_as_wrong']}")
    print(f"Accuracy rate: {results['acc_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
