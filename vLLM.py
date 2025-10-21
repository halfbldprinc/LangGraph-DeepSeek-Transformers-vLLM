import torch
import re
import gc
from typing import List, Dict, Optional, TypedDict, Any
from dataclasses import dataclass
from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DeepSeekConfig:
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"  
    max_tokens: int = 2048  
    temperature: float = 0.7 
    device: str = "cpu"  # check if cuda then switch to GPU 

# Formatting messages for DeepSeek
class MessageFormatter:
    @staticmethod
    def textToprompt(messages: List[BaseMessage]) -> str:
        prompt = []
        for msg in messages:
            if isinstance(msg, SystemMessage): 
                prompt.append(f"### System:\n{msg.content}")
            elif isinstance(msg, HumanMessage): # User input 
                prompt.append(f"### User:\n{msg.content}")
            elif isinstance(msg, AIMessage):  # State here (History)
                prompt.append(f"### Assistant:\n{msg.content}")
        prompt.append("### Assistant:")  
        return "\n\n".join(prompt) 
    
    @staticmethod
    def cl_reponse(response: str) -> str:
        # if the responce contains any of these words, remove them
        s_keywords = [
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|im_start|>", "<|im_end|>",
            "### Assistant:", "### User:", "### System:"
        ]
        for s in s_keywords:
            response = response.replace(s, "")
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        return response.strip()

# The memory manager - makes sure we don't overload the AI's brain
class TokenManager:
    def __init__(self, tokenizer, max_context: int = 1024):  # have token size for request half for respoonce 
        self.tokenizer = tokenizer  
        self.max_context = max_context  
    
    def offload_con(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        current_length = len(self.tokenizer.encode(
            MessageFormatter.textToprompt(messages)
        ))
        if current_length <= self.max_context:
            return messages
        
        truncated = []
        if messages and isinstance(messages[0], SystemMessage):
            truncated.append(messages[0])  
            remaining = messages[1:]  
        else:
            remaining = messages
        temp_messages = []
        for msg in reversed(remaining):
            test_messages = truncated + temp_messages + [msg]
            test_length = len(self.tokenizer.encode(
                MessageFormatter.textToprompt(test_messages)
            ))

            if test_length <= self.max_context:
                temp_messages.insert(0, msg)  
            else:
                break  
        return truncated + temp_messages  

# The main brain of our operation - handles talking to DeepSeek
class DeepSeekLLM:
    # Virtual Language Model (vLLM) interface for DeepSeek
    def __init__(self, config: DeepSeekConfig):
        self.config = config  
        try:
            # Determine device_map and device placement
            use_cuda = config.device == "cuda" and torch.cuda.is_available()
            device_map = "auto" if use_cuda else None

            print(f"Loading model: {config.model_name}")
            print(f"Running on: {'cuda' if use_cuda else 'cpu'}")

            # load model with optional device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float32,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # If device_map was not used and CUDA is requested, move model explicitly
            if not device_map:
                target_device = "cuda" if use_cuda else config.device
                try:
                    self.model = self.model.to(target_device)
                except Exception:
                    # fallback to cpu
                    self.model = self.model.to("cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                

            self.token_manager = TokenManager(self.tokenizer, config.max_tokens - 512)
            print("AI is ready!")
            
        except Exception as e:
            print(f"Error loading the AI: {e}")
            raise RuntimeError(f"Failed to load DeepSeek: {e}") from e

    @property
    def _llm_type(self) -> str:
        return "deepseek" 
    
    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # tokenize and move tensors to the configured device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # tokinize the input text to numbers   UPPER Line
            # text size check 
            if inputs.input_ids.shape[1] > self.config.max_tokens - 512:
                inputs.input_ids = inputs.input_ids[:, -(self.config.max_tokens - 512):]
                if hasattr(inputs, 'attention_mask'):
                    inputs.attention_mask = inputs.attention_mask[:, -(self.config.max_tokens - 512):]
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=min(512, self.config.max_tokens - inputs["input_ids"].shape[1]),
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True if self.config.temperature > 0 else False
                )
            
            # convert tocken back to text
            # decode the newly generated tokens (skip special tokens)
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            response = MessageFormatter.cl_reponse(response)
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"Error DeepSeek failed: {e}")
            return f"Error, DeepSeek Failed: {str(e)}"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.generate(prompt, stop)

# lang graph state ds for chat history
class ChatState(TypedDict):
    messages: List[BaseMessage]  
    token_count: int

def create_chatbot(config: DeepSeekConfig):
    llm = DeepSeekLLM(config) 
    def process_message(state: ChatState) -> ChatState:
        #  need the chat to be statefull and remember past messages
        prompt = MessageFormatter.textToprompt(state["messages"])
        
        # Control input size 
        state["messages"] = llm.token_manager.offload_con(state["messages"])
        response = llm._call(prompt)
        # update history ( state )
        return {
            "messages": state["messages"] + [AIMessage(content=response)],
            "token_count": len(llm.tokenizer.encode(prompt + response))
        }

    # Linking Nodes to draw graph 
    graph = StateGraph(ChatState)
    graph.add_node("process", process_message)  
    graph.set_entry_point("process") 
    graph.add_edge("process", END)  
    return graph.compile() 


def interactive_chat():
    config = DeepSeekConfig()  
    chatbot = None  
    
    try:
        chatbot = create_chatbot(config)  
        
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Be concise.")
        ] # instruction for ai  
        
        print("\nWelcome to DeepSeek Chat! (Type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("\nYou: ").strip() 
                if user_input.lower() in ['quit', 'exit']:
                    break  
                if not user_input:
                    continue  
                    
                messages.append(HumanMessage(content=user_input)) 
                result = chatbot.invoke({
                    "messages": messages,
                    "token_count": 0  
                })
                
                messages = result["messages"]
                print(f"\nAssistant: {messages[-1].content}")  
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                
    except Exception as e:
        print(f"Failed to start the chat: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if chatbot:
            gc.collect()

if __name__ == "__main__":
    interactive_chat() 