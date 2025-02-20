import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import numpy as np
import csv
import os

# Setting absolute paths for output, logs, and checkpoint
base_dir = os.path.join(os.path.dirname(__file__), '..')
output_dir = os.path.join(base_dir, 'minihex_blue')
logs_dir = os.path.join(base_dir, 'logs')
checkpoint_dir = os.path.join(base_dir, 'mini_blue_test', 'checkpoint-4931640')
tokenizer_dir = os.path.join(base_dir, 'tokenizer')
dataset_path = os.path.join(base_dir, 'ZERO_blue_wins.txt')

# Ensuring the directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Loading the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

# Loading the model from the checkpoint
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)

# Reading the new dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return Dataset.from_dict({'text': lines})

# Loading the new dataset
dataset = load_data(dataset_path)

# Splitting the dataset into training and validation sets (5% for validation)
train_size = int(0.95 * len(dataset))
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))

# Defining the function for tokenization
def tokenize_function(examples):
    # Adding special tokens to each text example
    text_with_special_tokens = ['<|startoftext|>' + text + '<|endoftext|>' for text in examples['text']]
    return tokenizer(text_with_special_tokens, padding='max_length', truncation=True, max_length=130)

# Tokenizing the data
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Setting up an adaptive batch size using data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Callback to save average loss per epoch in a CSV file
class SaveLossCallback(TrainerCallback):
    def __init__(self, output_file, trainer):
        self.output_file = output_file
        self.trainer = trainer
        self.epoch_train_losses = []
        # Creating the file with headers
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'average_training_loss', 'validation_loss', 'train_eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.epoch_train_losses.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        # Calculating the average training loss for the epoch
        avg_train_loss = np.mean(self.epoch_train_losses) if self.epoch_train_losses else None
        self.epoch_train_losses = []  # Resetting the list for the next epoch

        # Obtaining the validation loss
        val_results = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
        val_loss = val_results['eval_loss']

        # Calculating eval_loss for the training data
        print(f"Evaluating train loss after epoch {state.epoch}")
        train_eval_results = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        train_eval_loss = train_eval_results['eval_loss']

        # Writing results to the CSV file
        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([state.epoch, avg_train_loss, val_loss, train_eval_loss])

        return control

# Training parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=60,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir=logs_dir,
    logging_steps=100,  # Increased logging frequency for more accurate averaging
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=100,
    eval_accumulation_steps=10,
    save_steps=10000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Setting up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

# Adding the callback to save the loss
trainer.add_callback(SaveLossCallback(os.path.join(logs_dir, 'loss_log_finetune_blue.csv'), trainer))

# Training the model (fine-tuning)
trainer.train()