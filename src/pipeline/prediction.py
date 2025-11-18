from transformers import AutoTokenizer,AutoModelForSequenceClassification
from src.config.confriguration import ConfriguationManager
import torch
import json
import torch.nn.functional as F
from api_calls.recheck import verify_with_vertexai,extract_json


class PredictionPipelines:
    def __init__(self):
        self.config = ConfriguationManager()
        model_configs = self.config.get_model_evaluation_config()

        device="cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_configs.tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_configs.model_path).to(device)

        self.model.eval()


    def get_output(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class_id].item()


        labels = ["Fake","true"]
        predicted_label = labels[predicted_class_id]


        resp = verify_with_vertexai(text)
        ans = resp['label']

    
    
    
        if str(ans).lower() == str(predicted_label).lower():
            return {"label": predicted_label, "probability": round(confidence * 100, 2)}

        if str(ans).lower !="error":
            return {"label": ans, "probability": 100}
        else:
            return {"label":"Unable to find any information about it","probability":0}

