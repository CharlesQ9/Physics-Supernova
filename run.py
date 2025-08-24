import os

# Load environment variables from .env file if it exists
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

# use openrouter by default; if you want to use other API bases (e.g. openai api base, etc.), simply set OPENROUTER_API_BASE to your base
os.environ["OPENROUTER_API_BASE"] = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

import argparse

# for type hints
from typing import List
from smolagents.default_tools import Tool
from smolagents import (
    MultiStepAgent,
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)

# deal with markdown contents
from utils import MarkdownMessage
from utils import load_markdown_from_filepath, markdown_to_plaintext, markdown_images_compress

# Import tools usable for agents
from utils import WolframAlphaTool, AskImageTool, ReviewRequestTool, SummarizeMemoryTool


TOOLNAME2TOOL = {
    'wolfram_alpha_query': WolframAlphaTool,
    'ask_image_expert': AskImageTool,
    'ask_review_expert': ReviewRequestTool,
    'finalize_part_answer': SummarizeMemoryTool
}


def _create_Physics_agent(Tools_list:List[type[Tool]],
                         markdown_content: MarkdownMessage,
                         model_id: str = "openrouter/google/gemini-2.5-pro",
                         managed_agents_list: List[MultiStepAgent] = None,
                         max_steps:int=80,
                         **kwargs)-> ToolCallingAgent | CodeAgent: # either returns ToolCallingAgent or CodeAgent

    
    # LLM to use for the agent
    model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base=os.environ.get("OPENROUTER_API_BASE"),
        max_completion_tokens=32768,
        num_retries=3,
        timeout=1200,        
    )
    
    # tools for the manager agent
    tools = []
    for tool in Tools_list:
        if tool.__name__ == "AskImageTool" and "image_tool_model" in kwargs:
            tools.append(tool(vision_model_id=kwargs["image_tool_model"]))
        elif tool.__name__ == "ReviewRequestTool" and "review_tool_model" in kwargs:
            tools.append(tool(review_tool_model=kwargs["review_tool_model"]))
        elif tool.__name__ == "SummarizeMemoryTool" and "summarize_tool_model" in kwargs:
            tools.append(tool(summarize_model_id=kwargs["summarize_tool_model"]))
        else:
            tools.append(tool())
    
    
    # initialize the manager agent
    manager_agent_kwargs = dict(
        model=model,
        tools=tools,
        max_steps=max_steps,
        verbosity_level=2,
        name="physics_agent",
        description="",
        managed_agents=managed_agents_list
    )
    
    if kwargs["manager_type"] == "CodeAgent":
        manager_agent_kwargs["additional_authorized_imports"] = [
            "os", "sys", "time", "argparse", "pathlib",
            "matplotlib.pyplot", "numpy", "pandas"
        ]
        managerAgent = CodeAgent(**manager_agent_kwargs)
    elif kwargs["manager_type"] == "ToolCallingAgent":
        manager_agent_kwargs["max_tool_threads"] = 1
        managerAgent = ToolCallingAgent(**manager_agent_kwargs)
    else:
        raise ValueError(f"Unknown manager type: {kwargs['manager_type']}. Must be 'ToolCallingAgent' or 'CodeAgent'.")


    # register the tools with the agent
    for toolName in managerAgent.tools:
        managerAgent.tools[toolName].worker_agent = managerAgent


    # set high res images in the agent, for the AskImageTool to use
    managerAgent.markdown_content_high_res_image = markdown_content
    
    return managerAgent

def get_managed_agents_list(managed_agents_list: List[str] = None, managed_agents_list_model_id: str = None) -> List[MultiStepAgent]:
    if managed_agents_list is None:
        return []
    
    managed_agents = []
    for agent_name in managed_agents_list:
        # LLM model
        model = LiteLLMModel(
            model_id=managed_agents_list_model_id,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            api_base=os.environ.get("OPENROUTER_API_BASE"),
            max_completion_tokens=32768,
            num_retries=3,
            timeout=1200,        
        )
        # managed agent
        managed_agent = CodeAgent(
            tools=[],
            model=model,
            name=agent_name,
            additional_authorized_imports=["os", "sys", "time", "argparse", "pathlib","matplotlib.pyplot", "numpy", "pandas"],
            description=f"I am a managed agent with name {agent_name}. I can assist with code-related tasks",
            max_steps=80,
            verbosity_level=2,
        )
        managed_agents.append(managed_agent)
    
    return managed_agents

# create the agent
def create_agent(model_id:str = "openrouter/google/gemini-2.5-pro",
                        input_markdown_file:str=None,
                        tools_list: List[str]=[],
                        managed_agents_list: List[str] = None,
                        managed_agents_list_model_id:str=None,
                        **kwargs) -> ToolCallingAgent | CodeAgent:
    markdown_content = load_markdown_from_filepath(input_markdown_file)
    
    # create the manager agent
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]
    physicsAgent = _create_Physics_agent(Tools_list=ToolsList,
                                         markdown_content=markdown_content, model_id=model_id,
                                         managed_agents_list = get_managed_agents_list(managed_agents_list, managed_agents_list_model_id),
                                         **kwargs)
    return physicsAgent

# obtain task string and images for the agent to run.
def obtain_task_and_images(input_markdown_file:str=None,
                        tools_list: List[str]=[],
                        managed_agents_list: List[str] = None,
                        manager_type: str = "ToolCallingAgent",
                        ) -> tuple[str, List[bytes]]:
    
    # load markdown content with high res images
    markdown_content = load_markdown_from_filepath(input_markdown_file)
    
    # create the manager agent
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]
    
    
    # get task and images (to parse into agents) from the markdown content
    problem_text = markdown_to_plaintext(markdown_content)
    compressed_problem_images = markdown_images_compress(markdown_content, max_short_side_pixels=1080)
    
    # system prompt for the agent.
    task="Below is the full physics problem. If there are Images, Images are attached; reference them using their placeholders (e.g. <image_1>, <image_2>). "
    
    # system prompts for agent related to tools (if there are any)
    IMG_TOOL_PROMPT = "When you need to perform measurements on images, you MUST call the `ask_image_question` tool. EVERYTIME you MEASURE from some FIGURE, e.g., reading numbers, getting readings of items on figures, you MUST call the `ask_image_question` tool with the image reference and your question, or you might get very wrong measurements!"
    REVIEW_TOOL_PROMPT = "When you need expert review of your work, you MUST call the `ask_review_expert` tool. Your task is to solve the problem and use the reviewer expert when necessary to improve your answer until it's satisfactory or you reach 3 review iterations for the same small details."
    if SummarizeMemoryTool in ToolsList:
        REVIEW_TOOL_PROMPT += "Before you use the `finalize_part_answer` tool, you MUST use the `ask_review_expert` tool to review your (part) answer, to ensure that your answer is correct and complete."
    WOLFRAMALPHA_TOOL_PROMPT = "If you are provided with wolfram_alpha_query tools, When you need mathematical calculations, equation solving, or scientific computations, you can use the `wolfram_alpha_query` tool for accurate results."
    FINALIZE_PART_ANSWER_PROMPT = "When you are sure that you have finished a part of the problem, you should call the `finalize_part_answer` tool to summarize your work on that part and copy that to the answer sheet, including process and answer to these sub-problem (that you have finished till now), and write to the answer sheet. You should call this tool Only When you are sure that you have finished a Part of the problem."
    if ReviewRequestTool in ToolsList:
        FINALIZE_PART_ANSWER_PROMPT += "If you can use the review tool, you MUST use `ask_review_expert` tool to review your (part) answer before calling this tool, to ensure that your answer is correct and complete. You should call this tool only when you are sure that you have finished a part of the problem, and you have used the ask-review_expert tool! Also, you should not call it when there are parts left to solve!"
    task += IMG_TOOL_PROMPT if AskImageTool in ToolsList else ""
    task += REVIEW_TOOL_PROMPT if ReviewRequestTool in ToolsList else ""
    task += WOLFRAMALPHA_TOOL_PROMPT if WolframAlphaTool in ToolsList else ""
    task += FINALIZE_PART_ANSWER_PROMPT if SummarizeMemoryTool in ToolsList else ""
    
    
    # system propmts for agent related to managed agent(s) (if there are any)
    MANAGE_AGENT_PROMPT = f"You may use the managed Code Agent: {managed_agents_list} to assist you with code-related tasks."
    SELF_IS_CODE_AGENT_PROMPT = "You can use Python Code to execute programs, which may help with your task-solving process."
    task += MANAGE_AGENT_PROMPT if len(managed_agents_list) > 0 else ""
    task += SELF_IS_CODE_AGENT_PROMPT if manager_type == "CodeAgent" else ""
    
    # system prompts for agent related to the problem solving
    PROBLEM_SOLVING_PROMPT = (
            "Your task is to solve the problem part by part, step by step. "
            "ONLY after you have FISHED the WHOLE PROBLEM should you call final_answer, never call final_answer when there are parts left! Or the program will shut down immediately, and you would have NO CHANCE to continue solving!\n\n"
            "PROBLEM STATEMENT (text with image placeholders):\n"+ problem_text
    )
    task += PROBLEM_SOLVING_PROMPT
    
    
    return task, compressed_problem_images



def parse_args():
    ap = argparse.ArgumentParser(description="Run the Physics Agent with specified tools and model.")
    ap.add_argument(
        "--input-markdown-file",
        type=str,
        required=True,
        help="Path to the markdown file containing the physics problem.",
    )
    ap.add_argument(
        "--manager-model",
        type=str,
        default="openrouter/google/gemini-2.5-pro",
        help="Model ID to use for the agent.",
    )
    
    
    # Choose between ToolCallingAgent and CodeAgent for the manager agent
    ap.add_argument(
        "--manager-type",
        type=str,
        default="ToolCallingAgent",
        choices=["ToolCallingAgent", "CodeAgent"],
        help="Type of agent to create (default: ToolCallingAgent).",
    )
    
    
    
    # Tools available for the manager agent.
    ap.add_argument(
        "--tools-list",
        type=str,
        nargs='*',
        default=["wolfram_alpha_query", "ask_image_expert", "ask_review_expert"],
        help="List of tool names to use in the agent.",
    )
    # LLM model ids for the tools
    ap.add_argument(
        "--image-tool-model",
        type=str,
        default="openrouter/google/gemini-2.5-pro",
        help="Model ID to use for the image model.",
    )
    ap.add_argument(
        "--review-tool-model",
        type=str,
        default="openrouter/google/gemini-2.5-pro",
        help="Model ID to use for the review model.",
    )
    ap.add_argument(
        "--summarize-tool-model",
        type=str,
        default="openrouter/google/gemini-2.5-pro",
        help="Model ID to use for the summarizer model.",
    )
    
    
    # agent names of managed agents.
    ap.add_argument(
        "--managed-agents-list",
        type=str,
        nargs='*',
        default=[],
        help="List of managed agents to use in the agent.",
    )
    # LLM model ids for the managed agents
    ap.add_argument(
        "--managed-agents-list-model",
        type=str,
        default="openrouter/google/gemini-2.5-pro",
        help="Model ID to use for the agent.",
    )
    
    args = ap.parse_args()
    
    # for example, python run.py --input-markdown-file ./markdowns/physics_problem.md --manager-model openrouter/google/gemini-2.5-pro --tools-list wolfram_alpha_query ask_image_expert ask_review_expert
    if not args.input_markdown_file:
        raise ValueError("You must provide a markdown file position with --input-markdown-file.")
    if not os.path.exists(args.input_markdown_file):
        raise FileNotFoundError(f"Markdown file {args.input_markdown_file} does not exist.")
    return args


def main():
    args = parse_args()
    print(f"Running Physics Agent with model: {args.manager_model}, tools: {args.tools_list}, markdown file: {args.input_markdown_file}")
    
    managerAgent = create_agent(
        model_id=args.manager_model,
        input_markdown_file=args.input_markdown_file,
        tools_list=args.tools_list,
        image_tool_model=args.image_tool_model,
        review_tool_model=args.review_tool_model,
        managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
        managed_agents_list_model_id=args.managed_agents_list_model if hasattr(args, 'managed_agents_list_model') else None,
        manager_type = args.manager_type,
    )
    
    task,compressed_problem_images = obtain_task_and_images(
        input_markdown_file=args.input_markdown_file, # markdown file with problem statement and images, will be loaded and parsed to tasks and images.
        tools_list=args.tools_list, # system prompt is related to tools
        managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None, # system prompt is related to managed agents
        manager_type = args.manager_type, # system prompt is related to manager type (ToolCallingAgent or CodeAgent
    )
    
    
    managerAgent.run(task, images=compressed_problem_images)
    

if __name__ == "__main__":
    main()