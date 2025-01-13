from langchain import hub
from langchain.chat_models import ChatModel

def initialize_prompt() -> hub.Prompt:
    """
    Load the RAG prompt from LangChain prompt hub.
    Returns:
        hub.Prompt: Initialized prompt object.
    """
    return hub.pull("rlm/rag-prompt")

def generate_response(llm: ChatModel, prompt: hub.Prompt, question: str, context: str) -> str:
    """
    Generate a response from the LLM based on the question and context.
    Args:
        llm (ChatModel): The initialized language model.
        prompt (hub.Prompt): The prompt object for RAG.
        question (str): User question.
        context (str): Context retrieved from documents.
    Returns:
        str: Generated response.
    """
    messages = prompt.invoke({"question": question, "context": context})
    response = llm.invoke(messages)
    return response.content
