from datasets import load_dataset, DatasetDict
import os
from huggingface_hub import login 

HF_TOKEN = os.getenv("HF_API_KEY")

login(token=HF_TOKEN)

train_ds = load_dataset("json", data_files="train.json")['train']
test_ds = load_dataset("json", data_files="test.json")['train']
val_ds = load_dataset("json", data_files="valid.json")['train']

def process_dataset(examples):
    indices = []
    answers = []
    questions = []

    for idx, question, answer in zip(examples['idx'], examples['question'], examples['answer']):
        try:
            if isinstance(answer, list):
                ans = f"{answer[0]}"
                for i in answer[:-1]:
                    ans += f", {i}"
            else:
                ans = answer
        except:
            ans = ""
        answers.append(ans)
        indices.append(idx)
        questions.append(question)
    
    return {
        "idx": indices,
        "Question": questions,
        "Answer": answers
    }

processed_train = train_ds.map(process_dataset, 
                          batched=True,
                          batch_size=8,
                          remove_columns=[col for col in train_ds.column_names if col not in ["idx", "Question", "Answer"]])

processed_test = test_ds.map(process_dataset, 
                          batched=True,
                          batch_size=8,
                          remove_columns=[col for col in test_ds.column_names if col not in ["idx", "Question", "Answer"]])

processed_val = val_ds.map(process_dataset, 
                          batched=True,
                          batch_size=8,
                          remove_columns=[col for col in val_ds.column_names if col not in ["idx", "Question", "Answer"]])

dataset_dict = DatasetDict({
    "train": processed_train,
    "validation": processed_val,
    "test": processed_test
})

dataset_dict.push_to_hub("huyhoangt2201/Light-MedVLM-instruction-tuning-data")