import pandas as pd
import os
import re
from groq import Groq

def classify_with_regex(log_msg):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_msg):
            return label
    return None

def classify_with_llm(log_msg, groq_client):
    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        # model="llama-3.3-70b-versatile",
        model="deepseek-r1-distill-llama-70b",
        temperature=0.5
    )

    content = chat_completion.choices[0].message.content
    match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1).strip()
    return category

def classify_with_bert(log_msg, bert_model, classifier_model):
    embeddings = bert_model.encode([log_msg])
    probabilities = classifier_model.predict_proba(embeddings)[0]
    
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    predicted_label = classifier_model.predict(embeddings)[0]
    return predicted_label

def classify_logs(df: pd.DataFrame, bert_model, classifier_model, groq_client) -> list:
    """
    Main classification function that applies the hybrid logic to a DataFrame.
    """
    labels = []
    for index, row in df.iterrows():
        source = row["source"]
        log_msg = row["log_message"]
        
        if source == "LegacyCRM":
            label = classify_with_llm(log_msg, groq_client)
        else:
            label = classify_with_regex(log_msg)
            if not label:
                label = classify_with_bert(log_msg, bert_model, classifier_model)
        
        labels.append(label)
        
    return labels