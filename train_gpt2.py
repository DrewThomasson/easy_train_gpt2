import os
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Parameters
dataset_path = 'batman.csv'
output_dir = 'output'

# Load the dataset
data_files = {"train": dataset_path}
dataset = load_dataset('csv', data_files=data_files)

# Initialize the tokenizer and set the pad token
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Concatenate prompt and output into a single text entry for each example
    concatenated_examples = [p + tokenizer.eos_token + o for p, o in zip(examples['prompt'], examples['output'])]
    return tokenizer(concatenated_examples, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'output'])

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="no",  # No model saving during training
    save_total_limit=1,  # Maximum number of models to keep
    load_best_model_at_end=True,  # Optional: Load the best model at the end of training
    evaluation_strategy="no"  # Assuming no evaluation is performed
)

# Initialize the Data Collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Verify all files are saved
for root, dirs, files in os.walk(output_dir):
    for file in files:
        print(os.path.join(root, file))
