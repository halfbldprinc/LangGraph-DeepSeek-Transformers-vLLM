import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Download deepseek-coder-1.3b-instruct model if not already present (from huggin face) and change the path accordingly
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-instruct/snapshots/e063262dac8366fc1f28a4da0ff3c50ea66259ca")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

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
    input_ids = tokenizer(state["history"], return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Format response and update history (state)
    generated = response[len(state["history"]):].strip()
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

print("AI ready to chat  (Type 'exit' to quit)")
app.invoke({"history": "", "exit": False})
