from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import requests
from dotenv import load_dotenv
import traceback
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import json
from datetime import datetime

# Download necessary NLTK resources
try:
    nltk.download('punkt')
except:
    print("NLTK punkt download failed, but may already be installed")

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize embedding model for RAG
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load your trained models
def load_models():
    print("Loading models...")
    # Load tokenizer once (shared between models)
    tokenizer = T5Tokenizer.from_pretrained('../saved_emotion_model')
    
    emotion_model_path = "../saved_emotion_model"
    context_model_path = "../saved_context_model"

    emotion_model = T5ForConditionalGeneration.from_pretrained(emotion_model_path)
    context_model = T5ForConditionalGeneration.from_pretrained(context_model_path)

    emotion_model.eval()
    context_model.eval()
    print("Models loaded.")
    return emotion_model, context_model, tokenizer

# Initialize models
emotion_model, context_model, tokenizer = load_models()

# Initialize FAISS vector store with story content
def initialize_vectorstore():
    print("Initializing FAISS vector store...")
    # Read content from story.txt
    with open('./story.txt', 'r', encoding='utf-8') as f:
        story_lines = [line.strip() for line in f if line.strip()]  # remove empty lines

    # Create LangChain Document objects
    documents = [Document(page_content=line) for line in story_lines]

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(documents, embedding)
    print("FAISS vector store initialized.")
    return vectorstore

# Initialize vector store
vectorstore = initialize_vectorstore()

# Load reference responses from JSON file
def load_reference_responses():
    try:
        with open('reference_responses.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Initial reference responses based on your examples
        reference_responses = {
            "I pain": "I'm experiencing significant physical discomfort right now. My stomach has been hurting since after lunch, and the pain is becoming increasingly difficult to manage. I'm trying to find a comfortable position by leaning forward, but it's becoming overwhelming.",
            "Want eat": "I would like something to eat now. I'm particularly craving one of my favorite peanut butter sandwiches - specifically the way I enjoy them folded in half, which makes them easier for me to handle and eat.",
            "I tired": "I'm feeling extremely fatigued at the moment. The combination of the bright lights and loud noises in this room is overwhelming my senses. I need some time to rest in a calmer environment.",
            "I drink": "I would like something to drink now. Having a beverage is comforting to me, similar to how I feel when sharing genuine moments of laughter and connection with others.",
            "Want help": "I need assistance right now. I'm feeling overwhelmed by the sensory input from the bright lights and noise in this room. After a long day, I'm finding it difficult to cope with everything on my own and would appreciate some support."
        }
        # Save the initial reference responses
        with open('reference_responses.json', 'w', encoding='utf-8') as f:
            json.dump(reference_responses, f, indent=4)
        return reference_responses

# Load reference responses
reference_responses = load_reference_responses()

def get_mistral_response(prompt):
    print("Calling Mistral API...")
    api_key = "icX254HahAo8hdXKlhwPRyuoSNQbSCMh"
    # api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("MISTRAL_API_KEY not found in environment variables")
        raise ValueError("MISTRAL_API_KEY not found in environment variables")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=data
    )
    print(f"Mistral API status: {response.status_code}")
    print(f"Mistral API response: {response.text}")

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error from Mistral API: {response.text}")

def predict_with_models(text):
    print(f"Predicting emotion and context for: {text}")
    # Emotion prediction
    emo_input = tokenizer("Detect emotion: " + text, return_tensors="pt")
    emo_input = {k: v.to(emotion_model.device) for k, v in emo_input.items()}
    emo_output = emotion_model.generate(**emo_input, max_new_tokens=10)
    emotion = tokenizer.decode(emo_output[0], skip_special_tokens=True)
    print(f"Predicted emotion: {emotion}")

    # Context prediction
    ctx_input = tokenizer("Detect context: " + text, return_tensors="pt")
    ctx_input = {k: v.to(context_model.device) for k, v in ctx_input.items()}
    ctx_output = context_model.generate(**ctx_input, max_new_tokens=10)
    context = tokenizer.decode(ctx_output[0], skip_special_tokens=True)
    print(f"Predicted context: {context}")

    return emotion, context

def get_rag_context(query):
    print(f"Retrieving RAG context for query: {query}")
    query_vector = embedding.embed_query(query)
    retrieved_docs = vectorstore.similarity_search_by_vector(query_vector, k=1)
    if retrieved_docs:
        print(f"RAG context found: {retrieved_docs[0].page_content}")
        return retrieved_docs[0].page_content
    else:
        print("No relevant RAG context found.")
        return "No relevant context found."

def calculate_metrics(generated_text, reference_text):
    """Calculate various NLP evaluation metrics between generated and reference texts."""
    metrics = {}
    
    try:
        # BLEU Score
        reference_tokens = [nltk.word_tokenize(reference_text.lower())]
        hypothesis_tokens = nltk.word_tokenize(generated_text.lower())
        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smooth)
        metrics['bleu'] = bleu_score
    except Exception as e:
        print(f"Error calculating BLEU: {str(e)}")
        metrics['bleu'] = 0.0
    
    try:
        # ROUGE Scores
        rouge = Rouge()
        rouge_scores = rouge.get_scores(generated_text, reference_text)[0]
        metrics['rouge1_f'] = rouge_scores['rouge-1']['f']
        metrics['rouge2_f'] = rouge_scores['rouge-2']['f']
        metrics['rougeL_f'] = rouge_scores['rouge-l']['f']
    except Exception as e:
        print(f"Error calculating ROUGE: {str(e)}")
        metrics['rouge1_f'] = 0.0
        metrics['rouge2_f'] = 0.0
        metrics['rougeL_f'] = 0.0
    
    try:
        # BERTScore (need to run this on a GPU environment ideally)
        # Note: If this is failing or taking too long, you might want to disable it
        P, R, F1 = bert_score([generated_text], [reference_text], lang="en")
        metrics['bert_score_f1'] = F1.item()
    except Exception as e:
        print(f"Error calculating BERTScore: {str(e)}")
        metrics['bert_score_f1'] = 0.0
    
    return metrics

@app.route('/api/reference', methods=['GET', 'PUT'])
def manage_references():
    """Endpoint to get or update reference responses"""
    global reference_responses
    
    if request.method == 'GET':
        return jsonify(reference_responses)
    
    elif request.method == 'PUT':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update reference responses
        reference_responses.update(data)
        
        # Save updated references
        with open('reference_responses.json', 'w', encoding='utf-8') as f:
            json.dump(reference_responses, f, indent=4)
        
        return jsonify({
            'status': 'success',
            'message': 'Reference responses updated',
            'references': reference_responses
        })

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        if not data or 'input' not in data:
            print("No input provided in request.")
            return jsonify({'error': 'No input provided'}), 400

        text = data['input']
        print(f"Processing input: {text}")
        
        # Get predictions from both models
        emotion, context = predict_with_models(text)
        
        # Get RAG context using combined emotion and context as query
        query = emotion + " " + context
        rag_context = get_rag_context(query)
        
        # Format for prompt
        prompt = f"""
You are an expressive voice for a non-speaking AAC (Augmentative and Alternative Communication) user.
Your task is to expand their short or fragmented inputs into full, expressive statements AS IF YOU ARE THE USER speaking.
Do not respond as a conversational assistant talking TO the user.

The user is feeling {emotion}.

IMPORTANT: Use the provided personal context information to personalize the expanded statement when relevant:
User context: {rag_context}

**Guidelines:**
* Transform telegraphic or simple inputs into natural first-person statements
* Express the user's likely meaning, needs, emotions, or thoughts
* Use "I" statements and speak directly AS the AAC user
* Keep the expanded message authentic to the user's original intent
* Never reply with questions or responses back to the user
* Always maintain the user's perspective and agency
* When applicable, subtly incorporate the provided context information to personalize the message
* Ensure the personalization feels natural and not forced

Now expand the following user input:

Input: "{text}"
"""
        print(f"Prompt sent to Mistral:\n{prompt}")
        response = get_mistral_response(prompt)
        print(f"Mistral response: {response}")

        # Calculate metrics if a reference exists for this input
        metrics = {}
        if text in reference_responses:
            metrics = calculate_metrics(response, reference_responses[text])
            print(f"Evaluation metrics: {metrics}")

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to dialogue history log
        with open("dialogue_history.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"User Input: {text}\n")
            log_file.write(f"Emotion: {emotion}, Context: {context}\n")
            log_file.write(f"Generated Response: {response}\n")
            if metrics:
                log_file.write(f"Metrics: {json.dumps(metrics)}\n")
            log_file.write("\n")

        # Also append to a structured JSON log
        try:
            with open("dialogue_metrics.json", "r", encoding="utf-8") as json_file:
                try:
                    log_data = json.load(json_file)
                except json.JSONDecodeError:
                    log_data = []
        except FileNotFoundError:
            log_data = []
            
        log_entry = {
            "timestamp": timestamp,
            "input": text,
            "emotion": emotion,
            "context": context,
            "response": response,
            "metrics": metrics
        }
        log_data.append(log_entry)
        
        with open("dialogue_metrics.json", "w", encoding="utf-8") as json_file:
            json.dump(log_data, json_file, indent=4)

        return jsonify({
            'response': response,
            'emotion': emotion,
            'context': context,
            'metrics': metrics,
            'status': 'success'
        })

    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)