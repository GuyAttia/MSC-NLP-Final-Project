import math
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator

from modeling.rumour_eval_dataset_bert import RumourEval2019Dataset_BERTTriplets
from plot_results import plot_array_values_against_length, plot_confusion_matrix

from utils.utils import count_parameters, get_class_weights
from collections import Counter, defaultdict
from typing import Callable, Tuple, List


class BERT_Framework:

    def __init__(self, config: dict):
        self.config = config
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.config["variant"], cache_dir="./.BERTcache",
                                                       do_lower_case=True)

    def create_dataset_iterators(self):
        # Create DataSets
        fields = RumourEval2019Dataset_BERTTriplets.prepare_fields_for_text()
        train_data = RumourEval2019Dataset_BERTTriplets(self.config["train_data"], fields, self.tokenizer,
                                                        max_length=self.config["hyperparameters"]["max_length"])
        dev_data = RumourEval2019Dataset_BERTTriplets(self.config["dev_data"], fields, self.tokenizer,
                                                      max_length=self.config["hyperparameters"]["max_length"])
        test_data = RumourEval2019Dataset_BERTTriplets(self.config["test_data"], fields, self.tokenizer,
                                                       max_length=self.config["hyperparameters"]["max_length"])

        # Create iterators
        train_iter = BucketIterator(train_data, sort_key=lambda x: -len(x.text), sort=True,
                                    shuffle=False,
                                    batch_size=self.config["hyperparameters"]["batch_size"],
                                    device=self.device)
        create_non_repeat_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.text), sort=True,
                                                            shuffle=False,
                                                            batch_size=self.config["hyperparameters"]["batch_size"],
                                                            device=self.device)
        
        dev_iter = create_non_repeat_iter(dev_data)
        test_iter = create_non_repeat_iter(test_data)

        print(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        # Calculate weights for current data distribution
        weights = get_class_weights(train_data.examples, "stance_label", 4)

        return train_iter, dev_iter, test_iter, weights

    def fit(self, modelfunc: Callable, lr:int=None) -> dict:
        
        # Init counters and flags
        config = self.config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_losses, train_accuracies, train_F1s_global, train_F1s_weighted = [], [], [], []
        validation_losses, validation_accuracies, validation_F1s_global, validation_F1s_weighted = [], [], [], []
        test_losses, test_accuracies, test_F1s_global, test_F1s_weighted = [], [], [], []
        best_val_loss, best_val_acc, best_val_F1 = math.inf, 0, 0
        test_accuracy = 0
        best_val_loss_epoch = -1     

        train_iter, dev_iter, test_iter, weights = self.create_dataset_iterators()

        model = modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache").to(self.device)

        if lr:
            optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr)
        else:
            optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"])
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))

        for epoch in range(config["hyperparameters"]["epochs"]):
            self.epoch = epoch

            # train model on training examples
            train_loss, train_acc, train_F1_global, train_F1_weighted= self.train(model, lossfunction, optimizer, train_iter, config)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_F1s_global.append(train_F1_global)
            train_F1s_weighted.append(train_F1_weighted)

            # validate model on validation set
            validation_loss, validation_acc, validation_F1_global, validation_F1_weighted= self.validate(model, lossfunction, dev_iter, config)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_acc)
            validation_F1s_global.append(validation_F1_global)
            validation_F1s_weighted.append(validation_F1_weighted)

            # save loss/accuracy/F1 metrics in case of improvement
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_val_loss_epoch = epoch

            if validation_acc > best_val_acc:
                best_val_acc = validation_acc

            if validation_F1_global > best_val_F1:
                best_val_F1 = validation_F1_global
                # calculate metrics on test set
            test_loss, test_accuracy, F1_test_global, test_F1_weighted = self.validate(model, lossfunction, test_iter, config)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_F1s_global.append(F1_test_global)
            test_F1s_weighted.append(test_F1_weighted)

            # print results
            print(f"Epoch: {epoch}")
            print(f"Train- loss: {train_loss}, accuracy: {train_acc}")
            print(f"Validation- loss: {validation_loss}, accuracy: {validation_acc}, F1: {validation_F1_global}\n"
                  f"(Best loss: {best_val_loss} Best accuracy: {best_val_acc}, Best F1: {best_val_F1})")
            print(f"Test- accuracy: {test_accuracy}, F1: {F1_test_global}")

            # early stopping 
            if validation_loss > best_val_loss and epoch > best_val_loss_epoch + self.config["early_stop_after"]:
                print("Early stopping...")
                break
            
        if config["plot_res"] == "True":
            plot_array_values_against_length([train_losses, validation_losses, test_losses], "Loss vs Epochs")
            plot_array_values_against_length([train_accuracies, validation_accuracies, test_accuracies], "Accuracy vs Epochs")
            plot_array_values_against_length([train_F1s_global, validation_F1s_global, test_F1s_global], "Global F1 score vs Epochs")
            plot_array_values_against_length([train_F1s_weighted, validation_F1s_weighted, test_F1s_weighted], "Weighted F1 score vs Epochs")
            #plot_confusion_matrix(self.total_labels, self.total_preds)

    def train(self, model: torch.nn.Module, lossfunction: _Loss, optimizer: torch.optim.Optimizer,
              train_iter: Iterator, config: dict) -> Tuple[float, float]:

        # Init accumulators & flags
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        N = 0
        updated = False
        self.total_labels = []
        self.total_preds = []

        # In case of gradient accumulalation, how often should gradient be updated
        update_ratio = config["hyperparameters"]["true_batch_size"] // config["hyperparameters"]["batch_size"]

        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            updated = False
            pred_logits = model(batch)
            _, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
            loss = lossfunction(pred_logits, batch.stance_label) / update_ratio
            loss.backward()

            if (i + 1) % update_ratio == 0:
                optimizer.step()
                optimizer.zero_grad()
                updated = True

            # Update accumulators
            train_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.stance_label])
            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            self.total_preds += list(argmaxpreds.cpu().numpy())
            self.total_labels += list(batch.stance_label.cpu().numpy())

        # Do the last step if needed with what has been accumulated
        if not updated:
            optimizer.step()
            optimizer.zero_grad()

        loss = train_loss / N
        accuracy = total_correct / examples_so_far
        F1_global = metrics.f1_score(self.total_labels, self.total_preds, average="macro").item()  
        F1_weighted = metrics.f1_score(self.total_labels, self.total_preds, average='weighted').item()

        return loss, accuracy, F1_global, F1_weighted

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict) -> Tuple[float, float, float, List[float]]:

        train_flag = model.training
        model.eval()

        # init accumulators & flags
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        N = 0
        total_correct_per_level = Counter()
        total_labels = []
        total_preds = []

        for _, batch in enumerate(dev_iter):
            pred_logits = model(batch)
            loss = lossfunction(pred_logits, batch.stance_label)
            _, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)

            # compute branch statistics
            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]

            # compute correct and correct per branch depth
            correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label, levels=branch_levels)
            total_correct += correct
            total_correct_per_level += correct_per_level
            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.stance_label])
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(batch.stance_label.cpu().numpy())

        loss = dev_loss / N 
        accuracy = total_correct / examples_so_far

        F1_global = metrics.f1_score(total_labels, total_preds, average="macro").item()
        F1_weighted = metrics.f1_score(total_labels, total_preds, average='weighted').item()
        if train_flag:
            model.train()
        return loss, accuracy, F1_global, F1_weighted

    def calculate_correct(self, pred_logits: torch.Tensor, labels: torch.Tensor, levels=None):
        preds = torch.argmax(pred_logits, dim=1)
        correct_vec = preds == labels
        if not levels:
            return torch.sum(correct_vec).item()
        else:
            sums_per_level = defaultdict(lambda: 0)
            for level, correct in zip(levels, correct_vec):
                sums_per_level[level] += correct.item()
            return torch.sum(correct_vec).item(), sums_per_level
