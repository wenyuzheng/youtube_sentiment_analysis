import numpy as np
import os
import gc
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AdamW
import warnings
# pad_to_max_length is getting deprecated in transformers
warnings.filterwarnings("ignore")
from trainers.print_helpers import heading, inline_heading, status_update
from trainers.dataset import SST5Dataset, load_dataset

# Select the device to run training on
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CORE_COUNT = os.cpu_count()

class ModelTrainer:
  def __init__(self, learning_rate = 1e-5, max_length = 256, num_epochs = 1, batch_size = 32, seed_num = 123, sample_batch = None):
    self.learning_rate = learning_rate
    self.max_length = max_length
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.seed_num = seed_num
    self.sample_batch = sample_batch

  def seed(self):
    # For reproducability manually seed
    random.seed(self.seed_num)
    random.seed(self.seed_num)
    np.random.seed(self.seed_num)
    torch.manual_seed(self.seed_num)
    torch.cuda.manual_seed_all(self.seed_num)

  def load_model(self, model_name = 'bert-base-multilingual-cased'):
    self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    self.model = BertForSequenceClassification.from_pretrained(
      model_name,
      num_labels = 5,
      id2label = {0: "VERY NEGATIVE", 1: "SOMEWHAT NEGATIVE", 2: "NEUTRAL", 3: "SOMEWHAT POSITIVE", 4: "VERY POSITIVE"},
      label2id = {"VERY NEGATIVE": 0, "SOMEWHAT NEGATIVE": 1, "NEUTRAL": 2, "SOMEWHAT POSITIVE": 3, "VERY POSITIVE": 4},
      output_attentions = False,
      output_hidden_states = False
    )

  def load_optimizer(self):
    self.optimizer = AdamW(
      self.model.parameters(),
      lr = self.learning_rate,
      eps = 1e-8
    )

  def sst5_dataset(self):
    df_train = load_dataset('train')
    df_test = load_dataset('test')
    train_loader = DataLoader(SST5Dataset(df_train, self.encode_train), batch_size=self.batch_size, shuffle=True, num_workers=CORE_COUNT)
    test_loader = DataLoader(SST5Dataset(df_test, self.encode_train), batch_size=self.batch_size, shuffle=False, num_workers=CORE_COUNT)
    return train_loader, test_loader

  def train(self):
    self.seed()
    self.load_model()
    self.load_optimizer()

    # Load the data
    train_dataloader, test_dataloader = self.sst5_dataset()

    # Send the model to the device
    self.model.to(DEVICE)

    for epoch in range(0, self.num_epochs):
      inline_heading('Epoch {:} / {:}'.format(epoch + 1, self.num_epochs))

      # Enable gradient calculations
      torch.set_grad_enabled(True)
      # Set the total loss to 0 for this epoch
      self.train_loss = 0

      inline_heading('Started Training')
      # Set the model into train mode
      self.model.train()

      for i, batch in enumerate(train_dataloader):
        if self.sample_batch and i == self.sample_batch - 1:
          # Stop iterating when the sample batch limit is reached
          break
        self.process_training_batch(batch, i)

      heading('Training Loss: {:}'.format(self.train_loss))

      self.validate_model(test_dataloader)

      self.save_model()

      # Freeup memory between epochs
      gc.collect()
    # Release all the GPU memory cache that can be freed after finishing training
    torch.cuda.empty_cache()

  def process_training_batch(self, batch, batch_id):
    status_update('Training Batch: {:}'.format(batch_id))

    input_ids = batch[0].to(DEVICE)
    input_mask = batch[1].to(DEVICE)
    token_type_ids = batch[2].to(DEVICE)
    labels = batch[3].to(DEVICE)

    # Zero the gradients
    self.model.zero_grad()

    outputs = self.model(
      input_ids,
      token_type_ids=token_type_ids,
      attention_mask=input_mask,
      labels=labels
    )

    # Get the loss from the outputs tuple: (loss, logits)
    loss = outputs[0]

    # Convert the loss from a torch tensor to a number.
    # Calculate the total loss.
    self.train_loss = self.train_loss + loss.item()

    # Zero the gradients
    self.optimizer.zero_grad()

    # Perform a backward pass to calculate the gradients.
    loss.backward()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    # Optimizer to update the model weights:
    # Optimization step for GPUs
    if DEVICE.type == 'cuda':
        self.optimizer.step()

  def validate_model(self, test_dataloader):
    # This is supposed to be run as part of epoch
    inline_heading('Started Validation')

    # Set the model into evaluation mode, so we don't update the weights
    self.model.eval()

    # Turn off the gradient calculations.
    # This means the model will not compute or store gradients.
    # This step saves memory and speeds up validation.
    torch.set_grad_enabled(False)

    targets_list = []
    total_validation_loss = 0

    for j, batch in enumerate(test_dataloader):
      if self.sample_batch and j == ((self.sample_batch / 2) - 1):
        break

      status_update('Evaluation Batch: {:}'.format(j))

      input_ids = batch[0].to(DEVICE)
      input_mask = batch[1].to(DEVICE)
      token_type_ids = batch[2].to(DEVICE)
      labels = batch[3].to(DEVICE)

      outputs = self.model(
        input_ids,
        token_type_ids=token_type_ids,
        attention_mask=input_mask,
        labels=labels
      )

      # Tuple: (loss, logits)
      loss = outputs[0]
      # Convert from TorchTensor to a number and calculate the total loss.
      total_validation_loss = total_validation_loss + loss.item()
      # Predictions are the indices of the highest-scoring tokens
      predictions = outputs[1]
      # OPTIMIZATION: Move predictions to CPU
      opt_prediction = predictions.detach().cpu().numpy()
      # OPTIMIZATION: Move labels to CPU
      opt_target = labels.to('cpu').numpy()

      targets_list.extend(opt_target)

      if j == 0:
        stacked_opt_prediction = opt_prediction
      else:
        stacked_opt_prediction = np.vstack((stacked_opt_prediction, opt_prediction))

    # Validation accuracy
    y_expected = targets_list
    y_predicted = np.argmax(stacked_opt_prediction, axis=1)
    accuracy = accuracy_score(y_expected, y_predicted)

    heading('Validation Loss: {:}'.format(total_validation_loss))
    heading('Validation Accuracy: {:}'.format(accuracy))

  def save_model(self):
    # As huggingface transformers compat model
    self.model.save_pretrained('./sentiments/model-mbert')
    self.tokenizer.save_pretrained('./sentiments/model-mbert')
    # To save as torch model only
    # torch.save(self.model.state_dict(), 'model.pt')

  def encode_train(self, text, label):
    encoded_dict = self.tokenizer.encode_plus(
      text,                         # Eval target
      add_special_tokens = True,    # Ensure that special tokens are added
      max_length = self.max_length, # Max token length can take is 512
      pad_to_max_length = True,     # Pad the sequence to max length
      truncation=True,              # Truncate sequences to max length
      return_attention_mask = True, # Output attention masks
      return_tensors = 'pt',        # As pytorch tensors
    )
    padded_token_list = encoded_dict['input_ids'][0]
    # [1, 1, 1, ..., 0, 0] where 1 is the index of the token to be included
    attention_mask = encoded_dict['attention_mask'][0]
    token_type_ids = encoded_dict['token_type_ids'][0]
    target = torch.tensor(label)
    return (padded_token_list, attention_mask, token_type_ids, target)

# ModelTrainer().train()
