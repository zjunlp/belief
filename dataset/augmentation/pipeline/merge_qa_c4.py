import json
import random
import os

# Define paths
text_file_path = "/disk0/xuhaoming/confidence/c4_en_500.json"
qa_file_path = "/disk0/xuhaoming/confidence/dataset/selected_100_samples_verified_training_baseline_qa_new.json"
# qa_file_path = "/disk0/xuhaoming/confidence/dataset/selected_100_samples_verified_training_baseline_qa0.json"
output_file_path = "/disk0/xuhaoming/confidence/dataset/augmentation/result/merged_longqa_c4.json"

# Parameter k
K = 2

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    print(f"Loading data...")
    text_data = load_json(text_file_path)
    qa_data = load_json(qa_file_path)

    print(f"Loaded {len(text_data)} text samples and {len(qa_data)} QA samples.")

    new_qa_data = []

    for item in qa_data:
        # Randomly select K text samples
        selected_texts = random.sample(text_data, K)
        
        for text_item in selected_texts:
            # Prepend text to question
            text_content = text_item['text']
            original_question = item['question']
            new_question = f"{text_content}\n{original_question}"
            
            new_item = {
                "question": new_question,
                "answer": item['answer']
            }
            new_qa_data.append(new_item)

    print(f"Generated {len(new_qa_data)} new QA samples.")
    
    save_json(new_qa_data, output_file_path)
    print(f"Saved merged data to {output_file_path}")

if __name__ == "__main__":
    main()
