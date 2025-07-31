from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data: str, file_name: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(formatted_text)
    return f"Data successfully saved to {file_name}"

duck_search = DuckDuckGoSearchRun()
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_search = WikipediaQueryRun(api_wrapper=wiki_wrapper)

def combined_search(query: str) -> str:
    duck_result = duck_search.run(query)
    wiki_result = wiki_search.run(query)
    return f"Wikipedia Result:\n{wiki_result}\n\nDuckDuckGo Result:\n{duck_result}"

search_tool = Tool(
    name="search",
    func=combined_search,
    description="Search using both Wikipedia and DuckDuckGo for comprehensive info"
)

save_tool = Tool(
    name="save_to_txt_file",
    func=save_to_txt,
    description="Save structured research output to a text file"
)
