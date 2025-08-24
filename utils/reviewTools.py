import os
import time
from typing import Any, Dict, List
from smolagents.default_tools import Tool

# Import smolagents components for LLM model and message handling
from smolagents import (
    LiteLLMModel
)
from smolagents.models import ChatMessage, MessageRole
from .markdown_utils import MarkdownMessage, compressed_image_content

class ReviewRequestTool(Tool):
    """Physics expert reviewer to provide critical reviews and feedback on solution accuracy."""

    name = "ask_review_expert"
    description = (
        "Request expert review of your current work. Provide your solution and what you want reviewed. "
        "You should parse-in your current solution to-be-reviewed through 'my_solution' input (or the reviewer would not be able to see it)."
        "The reviewer will provide detailed feedback to help improve your answer."
    )
    inputs = {
        "my_solution": {
            "type": "string",
            "description": "Your current solution that needs review. This must be provided clearly and completely as the part for the review expert to review."
        },
        "my_note": {
            "type": "string", 
            "description": "What aspects to focus on (e.g., 'Check calculations and units', 'Verify logic', 'Overall review'), or your note/uncertain points/things you feel may go wrong."
        }
    }
    output_type = "string"

    def __init__(self, worker_agent=None, review_tool_model:str = "openrouter/google/gemini-2.5-pro"):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to main agent for accessing problem context
        api_key = os.environ.get("OPENROUTER_API_KEY")
        # Initialize review model with high token limit for detailed feedback
        self.review_model = LiteLLMModel(
            model_id=review_tool_model,
            api_key=api_key,
            api_base=os.environ.get("OPENROUTER_API_BASE"),
            max_completion_tokens=16000,
            num_retries=3,
            timeout=600,
        )

    def forward(self, my_solution: str, my_note: str) -> str:  # type: ignore[override]
        """Request expert review of solution with focus areas and return detailed feedback."""
        if not self.worker_agent or not hasattr(self.worker_agent, "markdown_content_high_res_image"):
            return "Error: No markdown content available."
        
        # Get compressed version of problem images for review context
        markdown_content: MarkdownMessage = compressed_image_content(self.worker_agent.markdown_content_high_res_image)
        
        # System prompt for critical physics review with emphasis on accuracy
        system_prompt = (
            "You are an uncompromising physics peer-reviewer. Your job is to find *every* logical, mathematical error in the worker's answer. "
            "Check dimensional consistency, missing steps, incorrect sign conventions, numerical mistakes, and unclear explanations. Focus especially on wrong answers, less on presentations."
            "Be extremely critical: if something is wrong, point it out and request clarification or correction. Mainly focus on errors that would lead to a wrong result, rather than focusing extremely on presentation or style."
            "It is possible that the worker's answer is not correct, so please be prepared to provide detailed feedback. The worker's answer contains some error, so you must check and point it out. Also, if the worker reads measurements from image, make sure to remind the worker that whenever it reads or measures from image, it uses the ask_image_expert tool, or the readings might be very inaccurate.\n"
        )
        
        # Create comprehensive review instruction with solution and focus areas
        review_instruction = (
            f"Please review the following solution:\n\n"
            f"WORKER'S SOLUTION:\n{my_solution}\n\n"
            f"WORKER'S NOTE: {my_note}\n\n"
            f"Please provide detailed feedback on correctness. "
            f"Point out any errors, wrong steps, focus more on correctness rather than presentation."
            f"The original problem follows:"
        )
        
        # Combine review instruction with original problem content (including images)
        combined_content : List[Dict[str, Any]] = [
            {"type": "text", "text": review_instruction}
        ] + markdown_content.content
        
        # Prepare messages for review model with system prompt and review request
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=combined_content),
        ]
        
        # Retry logic for robust review generation
        max_try = 3
        returned = None
        for _ in range(max_try):
            returned = self.review_model.generate(messages).content
            if returned is not None:
                returned = returned.strip()
                if returned != "":
                    return returned
            time.sleep(5)  # Wait before retry
            
        # Fallback responses for failed review attempts
        NO_RESPONSE_RETURN = "The reviewer is temporarily unavailable. Please act as a Physics Expert and review the previous solution yourself."
        if returned is None:
            return NO_RESPONSE_RETURN
        if returned == "":
            return NO_RESPONSE_RETURN
        return "Error when calling Review, please try later."
