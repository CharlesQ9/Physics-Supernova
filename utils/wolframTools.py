import os
import time
import requests
import xml.etree.ElementTree as ET
from smolagents.default_tools import Tool

# Get WolframAlpha API credentials from environment
APP_ID = os.getenv('WOLFRAM_ALPHA_APP_ID', None)  # Required for WolframAlpha API access

def query_wolfram_alpha(query: str, app_id = APP_ID) -> str:
    """Query WolframAlpha API and return formatted results."""
    base_url = 'http://api.wolframalpha.com/v2/query'
    params = {
        'input': query,
        'appid': app_id,
        'format': 'plaintext',
        'output': 'xml'  # Request XML output to parse with ElementTree
    }

    try:
        # Make API request to WolframAlpha
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Check for errors in the API response
        if root.get('success') == 'false':
            error_message = root.find('.//error/msg')
            if error_message is not None:
                return f"Wolfram Alpha API Error: {error_message.text}"
            else:
                return "Wolfram Alpha API Error: Unknown error."

        # Extract and format results from response pods
        result_text = []
        for pod in root.findall('.//pod'):
            title = pod.get('title')
            for subpod in pod.findall('.//subpod'):
                plaintext = subpod.find('plaintext')
                if plaintext is not None and plaintext.text:
                    result_text.append(f"{title}: {plaintext.text.strip()}")
        
        if not result_text:
            return "No meaningful results found from Wolfram Alpha."

        return "\n".join(result_text)  # Join all results with newlines

    except requests.exceptions.RequestException as e:
        return f"Network or API request error: {e}"
    except ET.ParseError as e:
        return f"Error parsing XML response: {e}"


class WolframAlphaTool(Tool):
    """Tool that queries WolframAlpha for mathematical calculations, scientific computations, and factual information."""
    name = "wolfram_alpha_query"
    description = "Query WolframAlpha for mathematical calculations, scientific information, factual data, and computational answers. Useful for solving equations, getting mathematical insights, unit conversions, and retrieving encyclopedic information."
    inputs = {
        "query": {
            "type": "string", 
            "description": "The query to send to WolframAlpha. Can be mathematical equations, scientific questions, factual queries, or computational requests."
        }
    }
    output_type = "string"
    
    def forward(self, query: str) -> str:
        # Commented debug code for logging agent memory
        # with open("wolfram_alpha_query.txt", "w") as f:
        #     f.write(str(self.worker_agent.write_memory_to_messages().copy()))
        """Query WolframAlpha and return the result with retry logic."""
        # Retry logic for robust API interaction
        max_try = 3
        for _ in range(max_try):
            
            try:
                result = query_wolfram_alpha(query)
                return result
            except Exception as e:
                print(f"Error querying WolframAlpha: {str(e)}")
                time.sleep(5)  # Wait before retry
        return "No response from WolframAlpha after multiple attempts."


if __name__ == "__main__":
    # Example usage for testing the WolframAlpha integration
    query = "Please tell me about Schodinger's equation"
    result = query_wolfram_alpha(query)
    print(result)