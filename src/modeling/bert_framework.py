import math
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from modeling.rumour_eval_dataset_bert import RumourEval2019Dataset_BERTTriplets
from utils.utils import count_parameters, get_class_weights
from collections import Counter, defaultdict
from typing import Callable, Tuple, List


class BERT_Framework:
    """
    Framework implementing BERT training with input pattern:
    [CLS]src post. prev post[SEP]target post[SEP]
    This is our best model, submitted to RumourEval2019 competition, referred to as BERT_{big} in the paper
    """

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
                                    batch_size=self.config["hyperparameters"]["batch_size"], repeat=False,
                                    device=self.device)
        create_noshuffle_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.text), sort=True,
                                                            shuffle=False,
                                                            batch_size=self.config["hyperparameters"]["batch_size"],
                                                            repeat=False,
                                                            device=self.device)
        dev_iter = create_noshuffle_iter(dev_data)
        test_iter = create_noshuffle_iter(test_data)

        print(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        # Calculate weights for current data distribution
        weights = get_class_weights(train_data.examples, "stance_label", 4)
        print("class weights")
        print(f"{str(weights.numpy().tolist())}")

        return train_iter, dev_iter, test_iter, weights

    def fit(self, modelfunc: Callable) -> dict:
        """
        Trains the model and executes early stopping
        :param modelfunc: model constructor
        :return statistics of trained model
        """
        config = self.config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_iter, dev_iter, test_iter, weights = self.create_dataset_iterators()

        model = modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache").to(self.device)
        print(f"Model has {count_parameters(model)} trainable parameters.")
        print(f"Manual seed {torch.initial_seed()}")

        optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"])
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))

        # Init counters and flags
        best_val_loss, best_val_acc, best_val_F1 = math.inf, 0, 0
        best_F1_loss, best_loss_F1 = 0, 0
        bestF1_testF1, bestF1_testacc = 0, 0
        bestF1_test_F1s = [0, 0, 0, 0]
        best_val_F1s = [0, 0, 0, 0]
        best_val_loss_epoch = -1

        for epoch in range(config["hyperparameters"]["epochs"]):
            self.epoch = epoch
            # this loss is computed during training, during active dropouts etc so it wont be similar to validation loss
            # but computing it second time over all training data is slow
            # You can call validate on train_iter if you wish to have proper training loss

            train_loss, train_acc = self.train(model, lossfunction, optimizer, train_iter, config)
            validation_loss, validation_acc, val_F1, _ = self.validate(model,
                                                                        lossfunction,
                                                                        dev_iter,
                                                                        config)

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_val_loss_epoch = epoch

            if validation_acc > best_val_acc:
                best_val_acc = validation_acc

            if val_F1 > best_val_F1:
                best_val_F1 = val_F1
                _, bestF1_testacc, bestF1_testF1, bestF1_test_F1s = self.validate(model,
                                                                                    lossfunction,
                                                                                    test_iter,
                                                                                    config)

            # info
            print(f"Epoch {epoch}, Training loss|acc: {train_loss:.6f}|{train_acc:.6f}")
            print(f"Epoch {epoch}, Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f} - "
                f"(Best {best_val_loss:.4f}|{best_val_acc:4f}|{best_val_F1})\n Best Test F1 - {bestF1_testF1}")

            if validation_loss > best_val_loss and epoch > best_val_loss_epoch + self.config["early_stop_after"]:
                print("Early stopping...")
                break
        
        return {
            "best_loss": best_val_loss,
            "best_acc": best_val_acc,
            "best_F1": best_val_F1,
            "bestF1_loss": best_F1_loss,
            "bestloss_F1": best_loss_F1,
            "bestACC_testACC": bestF1_testacc,
            "bestF1_testF1": bestF1_testF1,
            "val_bestF1_C1F1": best_val_F1s[0],
            "val_bestF1_C2F1": best_val_F1s[1],
            "val_bestF1_C3F1": best_val_F1s[2],
            "val_bestF1_C4F1": best_val_F1s[3],
            "test_bestF1_C1F1": bestF1_test_F1s[0],
            "test_bestF1_C2F1": bestF1_test_F1s[1],
            "test_bestF1_C3F1": bestF1_test_F1s[2],
            "test_bestF1_C4F1": bestF1_test_F1s[3]
        }

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

    def train(self, model: torch.nn.Module, lossfunction: _Loss, optimizer: torch.optim.Optimizer,
              train_iter: Iterator, config: dict) -> Tuple[float, float]:
        """
        :param model: model inherited from torch.nn.Module
        :param lossfunction:
        :param optimizer:
        :param train_iter:
        :param config:
        :param verbose: whether to print verbose outputs at stdout
        :return: train loss and train accuracy
        """
        # Initialize accumulators & flags
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        N = 0
        updated = False

        # In case of gradient accumulalation, how often should gradient be updated
        update_ratio = config["hyperparameters"]["true_batch_size"] // config["hyperparameters"]["batch_size"]

        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            updated = False
            pred_logits = model(batch)
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

        # Do the last step if needed with what has been accumulated
        if not updated:
            optimizer.step()
            optimizer.zero_grad()

        return train_loss / N, total_correct / examples_so_far

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict) -> Tuple[float, float, float, List[float]]:
        """

        :param model: model inherited from torch.nn.Module
        :param lossfunction:
        :param dev_iter:
        :param config:
        :param verbose:
        :param log_results: whether to print verbose outputs at stdout
        :return: validation loss, validation accuracy, validation accuracies per level, validation F1, per class F1s
        """

        train_flag = model.training
        model.eval()

        # initialize accumulators & flags
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        N = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)
        total_labels = []
        total_preds = []

        for _, batch in enumerate(dev_iter):
            pred_logits = model(batch)
            loss = lossfunction(pred_logits, batch.stance_label)
            _, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)

            # compute branch statistics
            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
            for branch_depth in branch_levels: 
                total_per_level[branch_depth] += 1

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

        loss, acc = dev_loss / N, total_correct / examples_so_far

        F1 = metrics.f1_score(total_labels, total_preds, average="macro").item()
        allF1s = metrics.f1_score(total_labels, total_preds, average=None).tolist()
        if train_flag:
            model.train()
        return loss, acc, F1, allF1s
