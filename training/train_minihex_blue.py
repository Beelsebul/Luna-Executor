import torch
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import numpy as np
import csv
import os


# Setting absolute paths for output and logs
base_dir = os.path.join(os.path.dirname(__file__), '..')
output_dir = os.path.join(base_dir, 'minihex_blue')
logs_dir = os.path.join(base_dir, 'logs')
tokenizer_dir = os.path.join(base_dir, 'tokenizer')
dataset_path = os.path.join(base_dir, 'gen_0_blue_wins.txt')


# Ensuring the directories are existing
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Loading the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

# Creating the GPT model with a custom configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=130,
    n_ctx=130,
    n_embd=128,
    n_layer=4,
    n_head=4,
    pad_token_id=tokenizer.pad_token_id
)
model = GPT2LMHeadModel(config)

# Reading the data
def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return Dataset.from_dict({'text': lines})

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

# Defining a callback to save average loss per epoch
class SaveLossCallback(TrainerCallback):
    def __init__(self, output_file, trainer):
        self.output_file = output_file
        self.trainer = trainer
        self.epoch_train_losses = []
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'average_training_loss', 'validation_loss', 'train_eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.epoch_train_losses.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        # Calculating the average training loss per epoch
        avg_train_loss = np.mean(self.epoch_train_losses) if self.epoch_train_losses else None
        self.epoch_train_losses = []  # Resetting the list for the next epoch

        # Obtaining the validation loss
        val_results = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
        val_loss = val_results['eval_loss']

        # Calculating eval_loss for the training data
        print(f"Evaluating train loss after epoch {state.epoch}")
        train_eval_results = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        train_eval_loss = train_eval_results['eval_loss']

        # Logging the results in the CSV file
        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([state.epoch, avg_train_loss, val_loss, train_eval_loss])

        return control

# Setting training parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=60,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir=logs_dir,
    logging_steps=100,  # Increasing the logging frequency for more precise averages
    save_strategy="epoch",  # Changed from save_steps to save_strategy
    evaluation_strategy="epoch",
    save_total_limit=100,
    eval_accumulation_steps=10,
    save_steps=10000,
    # Adding the strategy to save the best model
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

# Adding the callback after initializing the Trainer
trainer.add_callback(SaveLossCallback(os.path.join(logs_dir, 'loss_log_blue_agent_new.csv'), trainer))

# Training the model
trainer.train()
