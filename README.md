
## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/halfbldprinc/SampleLangGraph-DeepSeek-.git
cd SampleLangGraph-DeepSeek-
```

### 2. Create Virtual Environment

Create and activate a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Model (First Run)

You have to download the DeepSeek Coder 1.3B model from Hugging Face and change the path accordingly

**Note:** The default path to model is`~/.cache/huggingface/hub/` directory (approximately 2.6GB).

### 5. Run the Application
 
vLLM wrapper implmentation 

```bash
python vLLM.py
```

the classic implmentation

```bash
python main.py
```

## Usage

1. After running the application, you'll see:
   ```
   Loading tokenizer and model...
   AI ready to chat (Type 'exit' to quit)
   👤 You: 
   ```

2. Type your message and press Enter to chat with the AI.

3. The AI will respond with code explanations, programming help, or general conversation.

4. To exit the application, type any of the following:
   - `exit`
   - `quit` 
   - `bye`

