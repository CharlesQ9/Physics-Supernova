from openai import OpenAI
from loguru import logger
import json
import os


import re
import json
from typing import Optional, Union, List

def send_multimodal_message(
    api_key: str,
    text: str = None,
    messages: List = None,
    image_paths: Optional[Union[str, List[str], List[tuple], dict]] = None,
    model: str = "google/gemini-2.5-pro",
    stream: bool = False,
    return_token_consumption: bool = True,
    system_message = None,
    solve_prompt_position = None,
    log_cost_file = None,
    label = ""
) -> Union[str, dict]:
    """
    Send a multimodal message (text + optional image(s)) to an OpenRouter-compatible LLM.

    Args:
        api_key: Your OpenRouter API key.
        text: Text prompt to send.
        image_paths: Path(s) to images. Can be:
            - list[str] or list[(annotation, path)] (old style)
            - dict[placeholder:str -> path or (annotation, path)] (new style)
        model: Model name (e.g., 'google/gemini-2.5-pro').
        stream: Whether to stream the response.
        return_token_consumption: Whether to return token consumption.

    Returns:
        Response string from the model or a dict with response and token consumption.
    """
    from openai import OpenAI

    client = OpenAI(
            base_url=os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
            api_key=api_key
        )
    
    if text is not None:
    
        

        def encode_image_to_base64(path):
            import base64
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        content = []

        def append_content(item):
            if content and item["type"] == "text" and content[-1]["type"] == "text":
                content[-1]["text"] += item["text"]
            else:
                content.append(item)

        if isinstance(image_paths, dict):
            pattern = re.compile("|".join(map(re.escape, image_paths.keys())))
            pos = 0
            for match in pattern.finditer(text):
                start, end = match.span()
                placeholder = match.group()

                if start > pos:
                    append_content({"type": "text", "text": text[pos:start]})

                img_val = image_paths[placeholder]
                annotation, path = img_val if isinstance(img_val, tuple) else (None, img_val)

                if annotation:
                    append_content({"type": "text", "text": annotation})

                encoded = encode_image_to_base64(path)
                append_content({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

                pos = end

            if pos < len(text):
                append_content({"type": "text", "text": text[pos:]})

        else:
            append_content({"type": "text", "text": text})
            if image_paths:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                for item in image_paths:
                    annotation, path = item if isinstance(item, tuple) else (None, item)
                    if annotation:
                        append_content({"type": "text", "text": annotation})
                    encoded = encode_image_to_base64(path)
                    append_content({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    else:
        pass


    if messages is None:
        if system_message is not None:
            messages = [{"role": "system", "content": system_message}]
        else:
            messages = []

        messages.append({"role": "user", "content": content})
    else:
        assert text == None
    
    
    if solve_prompt_position is not None:
        with open(solve_prompt_position, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
            

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )

    full_response = ""
    if stream:
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full_response += delta
        usage = getattr(chunk, 'usage', None)
    else:
        full_response = response.choices[0].message.content
        usage = getattr(response, 'usage', None)
    
    messages.append({"role": "assistant", "content": full_response})
    
    if log_cost_file is not None:
        # make sure the directory exists.
        if os.path.dirname(log_cost_file) != '':
            os.makedirs(os.path.dirname(log_cost_file), exist_ok=True)
        with open(log_cost_file, "a", encoding="utf-8") as f:
            log_entry = {
                "model": model,
                "label": label,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 'Cannot determine for streaming',
                    "completion_tokens": usage.completion_tokens if usage else 'Cannot determine for streaming',
                    "total_tokens": usage.total_tokens if usage else 'Cannot determine for streaming'
                },
                "prev_text_samples": str(messages[-2]["content"])[:200]+'......'+str(messages[-2]["content"])[-200:] if len(messages) > 1 else "",
                
            }
            f.write(json.dumps(log_entry) + "\n")

    if return_token_consumption:
        result = {
            "response": full_response,
            "input_tokens": usage.prompt_tokens if (usage ) else 'Cannot determine for streaming',
            "output_tokens": usage.completion_tokens if (usage ) else 'Cannot determine for streaming',
            "total_tokens": usage.total_tokens if (usage ) else 'Cannot determine for streaming',
            "messages": messages
        }
        return result

    return full_response



if __name__ == "__main__":
    
    with open(r"/mnt/e/ipho25/theoryProblems/images/fD6BE79inLoPdOBsY7nTnmVG6TpsvlCZ6.svg", "r") as f:
        svg_content = f.read()
    
    prompt = f"Here is a figure of measured data. It's image figure is here: <figure_A>. The task is to find the x-axis value of three peeks of f-f_0 from this figure; Please zoom into the picture very carefully in order to get the x-axis of these three peeks. Please think step by step: you can identify the peek, then use the reference line to read the x-axis value of that peek."
    
    
    reply = send_multimodal_message(
        api_key=os.environ["OPENROUTER_API_KEY"],
        text=prompt,
        # text="Here is a paper about our experiment. ... Here is Figure A: <figure_A>; Here is Figure B: <figure_B> ... . In figure Jingzhe, what letters are on the figure?",
        image_paths={
            "<figure_A>": ('Figure Qinwei. The img in png format.',"/mnt/e/ipho25/theoryProblems/images/fD6BE79inLoPdOBsY7nTnmVG6TpsvlCZ6.png"),
            # "<figure_B>": ("Figure Jingzhe. The image after the experiment.", "/mnt/e/ipho25/theoryProblems/images/fig3.5.png"),
            # "<figure_C>": "/mnt/e/ipho25/theoryProblems/images/fig3.5.png",
            },
        model = 'google/gemini-2.5-flash',
        # model = 'openai/gpt-4.1',
        # model='anthropic/claude-sonnet-4',
        # model='qwen/qwen-vl-max',
        stream=True,
        return_token_consumption=True,
        system_message="You are an assistant that sees very clearly into one image."
    )
    # model='openai/gpt-4.1'
    # model='anthropic/claude-opus-4'
    # model='google/gemini-2.5-pro'
    # model='x-ai/grok-4'
    print("\nResponse:", reply)
