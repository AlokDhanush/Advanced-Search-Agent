from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import search_tool, save_tool
import json
import re

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

tool_map = {
    "search": search_tool,
    "save": save_tool,
}

# This will hold the full last Gemini-generated response
last_full_response = None

AGENT_PROMPT = """
You are a research assistant. You can take the following actions:
- "search": to find info from the internet
- "save": to save text to a .txt file

You must always return only JSON (without any extra explanation or markdown), like:
{
  "action": "search",
  "input": "quantum computing overview"
}

If the user wants to save the last detailed output you generated, return:
{
  "action": "save",
  "input": "previous response"
}
"""

def clean_response(raw_response: str) -> dict:
    """Cleans Gemini response and extracts JSON."""
    raw = raw_response.strip().lstrip("`").rstrip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("Could not extract JSON from response.")
    return json.loads(match.group())

def run_agent():
    global last_full_response
    while True:
        user_input = input("\nWhat do you want to research or do? (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        # Compose conversation for Gemini
        messages = [
            ("system", AGENT_PROMPT),
            ("user", user_input)
        ]
        gemini_input = [{"role": role, "content": content} for role, content in messages]

        try:
            # Get Gemini's tool use decision
            tool_response = llm.invoke(gemini_input).content
            print(f"\nGemini Tool Plan:\n{tool_response}")

            action_data = clean_response(tool_response)
            action = action_data.get("action")
            input_text = action_data.get("input")

            if action not in tool_map:
                print(f"Unknown action: {action}")
                continue

            # Take action based on the query
            if action == "search":
                tool_output = tool_map[action].func(input_text)

                response_prompt = [
                    {"role": "system", "content": "Give a detailed and informative answer using this result:"},
                    {"role": "user", "content": tool_output}
                ]
                full_response = llm.invoke(response_prompt).content
                last_full_response = full_response  
                print(f"\nGemini Answer:\n{full_response}")

            elif action == "save":
                if input_text.strip().lower() == "previous response":
                    if last_full_response is None:
                        print("No previous Gemini response to save.")
                        continue
                    result = tool_map[action].func(last_full_response)
                else:
                    result = tool_map[action].func(input_text)

                print(f"\nSave Tool Output:\n{result}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_agent()
