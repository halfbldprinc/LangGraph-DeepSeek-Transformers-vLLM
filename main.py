import os
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, END
from typing import TypedDict


logger = logging.getLogger(__name__)


def get_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_path: str, device: str):
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}. Please download the model and point --model-path to it.")

    logger.info("Loading tokenizer and model from %s", model_path)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.eval()
    model.to(device)
    return tokenizer, model


# Data structure for chat state (chat history)
class ChatState(TypedDict):
    history: str
    exit: bool
# when user input a text it will be added to history ( for state full chat)
def user_node(state: ChatState) -> ChatState:
    user_input = input("👤 You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        state["exit"] = True
        return state
    state["history"] += f"\n### Instruction:\n{user_input}\n### Response:\n"
    return state

def model_node(state: ChatState) -> ChatState:
    # Tokenize and move inputs to the model device
    device = next(model.parameters()).device
    inputs = tokenizer(state["history"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Format response and update history (state)
    generated = response[len(state["history"]):].strip() if response.startswith(state["history"]) else response.strip()
    print(f"🤖 Bot: {generated}")
    state["history"] += generated + "\n"
    return state

def should_continue(state: ChatState) -> str:
    if state.get("exit", False):
        return END
    return "model"

# Build graph
graph = StateGraph(ChatState)
graph.add_node("user", user_node)
graph.add_node("model", model_node)
graph.set_entry_point("user")
graph.add_conditional_edges("user", should_continue, {"model": "model", END: END})
graph.add_edge("model", "user")

app = graph.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="~/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-instruct/snapshots/e063262dac8366fc1f28a4da0ff3c50ea66259ca", help="Path to the downloaded model folder or HF identifier")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = get_device(args.device)

    try:
        global tokenizer, model
        tokenizer, model = load_model_and_tokenizer(args.model_path, device)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise

    print("AI ready to chat  (Type 'exit' to quit)")
    app.invoke({"history": "", "exit": False})


if __name__ == "__main__":
    main()
