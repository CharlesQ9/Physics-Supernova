from judge_answer.call_llm_utils import send_multimodal_message
import os
from argparse import ArgumentParser

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


import re
import json

def obtain_newContent_and_images_from_markdown(content, image_base_dir):
    # search for ![](*.png), ![](*.jpg), ![](*.jpeg) in the content.
    IMG_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
    image_pattern = re.compile(r'!\[\]\(([^)]+)\)')
    images = dict()
    new_content = content
    
    image_counter = 0
    for match in image_pattern.finditer(content):
        image_path = match.group(1)
        if image_path.endswith(IMG_FILE_EXTENSIONS):
            image_counter += 1
            full_image_path = os.path.join(image_base_dir, image_path)
            if os.path.exists(full_image_path):
                current_key = f"<image_{image_counter}>"
                images[current_key] = full_image_path
                new_content = new_content.replace(match.group(0), current_key)
            else:
                print(f"Image not found: {full_image_path}")
    return new_content, images
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Send a multimodal message to the OpenRouter API.")
    parser.add_argument("--markdown_file", type=str, default=r"examples/Problems/IPhO25/theory1/theory1.md")
    parser.add_argument("--image_base_dir", type=str, default=r"examples/Problems/IPhO25/theory1/")
    parser.add_argument("--output_markdown", type=str, default="output/run_direct/theory1_gemini-2.5-flash.md")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash")
    
    args = parser.parse_args()
    
    mkd_path = args.markdown_file
    with open(mkd_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    
    
    new_content, imgs = obtain_newContent_and_images_from_markdown(
        content=content,
        image_base_dir=args.image_base_dir
    )
    
    # print(imgs)
    new_content+="\n\nPlease solve for the whole problem, include Part A,B,C and D, in one response. You do not need to ask for my permission to continue, just automatically continue and solve them. Start your solution:"
    
    
    reply = send_multimodal_message(
        api_key=os.environ["OPENROUTER_API_KEY"],
        text=new_content,
        # model="openai/gpt-4.1-mini",
        model=args.model,
        image_paths=imgs,
        stream=True,
        return_token_consumption=True,
        system_message="You are an expert in solving Physics Olympiad problems. Please provide a detailed solution to the problem presented by the user. Please think step by step, and check your answers carefully (from Physics point of view) once you finish each part.",
        solve_prompt_position = None
    )
    
    NAME = args.output_markdown
    # create the directory if it doesn't exist
    dir_name = os.path.dirname(NAME)
    os.makedirs(dir_name, exist_ok=True)
    response_mkd = reply["response"]
    
    with open(NAME, "w", encoding="utf-8") as f:
        f.write(response_mkd)
    
    # print(reply)