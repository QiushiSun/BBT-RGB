import torch
import torch.nn as nn
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, accuracy_score
from transformers import RobertaTokenizer
from utils import hinge_loss


class TaskMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {}

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()

        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class SST2Metric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.args = args
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('bad', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('great', add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class YelpPMetric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.args = args
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('bad', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('great', add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class AGNewsMetric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.args = args
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('World', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Sports', add_special_tokens=False)[0]: 1,
            tokenizer.encode('Business', add_special_tokens=False)[0]: 2,
            tokenizer.encode('Technology', add_special_tokens=False)[0]: 3,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class DBPediaMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode("Company", add_special_tokens=False)[0]: 0,
            tokenizer.encode("EducationalInstitution", add_special_tokens=False)[0]: 1,
            tokenizer.encode("Artist", add_special_tokens=False)[0]: 2,
            tokenizer.encode("Athlete", add_special_tokens=False)[0]: 3,
            tokenizer.encode("OfficeHolder", add_special_tokens=False)[0]: 4,
            tokenizer.encode("MeanOfTransportation", add_special_tokens=False)[0]: 5,
            tokenizer.encode("Building", add_special_tokens=False)[0]: 6,
            tokenizer.encode("NaturalPlace", add_special_tokens=False)[0]: 7,
            tokenizer.encode("Village", add_special_tokens=False)[0]: 8,
            tokenizer.encode("Animal", add_special_tokens=False)[0]: 9,
            tokenizer.encode("Plant", add_special_tokens=False)[0]: 10,
            tokenizer.encode("Album", add_special_tokens=False)[0]: 11,
            tokenizer.encode("Film", add_special_tokens=False)[0]: 12,
            tokenizer.encode("WrittenWork", add_special_tokens=False)[0]: 13,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class MRPCMetric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.args = args
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('No', add_special_tokens=False)[0]: 0,  # not dumplicate
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 1,  # dumplicate
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        f1 = f1_score(self._target, self._pred, average='macro')
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'f1': f1,
                'hinge': hinge_loss,
                'ce': ce_loss}


class SNLIMetric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.args = args
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Maybe', add_special_tokens=False)[0]: 1,
            tokenizer.encode('No', add_special_tokens=False)[0]: 2,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class TRECMetric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.args = args
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('Description', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Entity', add_special_tokens=False)[0]: 1,
            tokenizer.encode('Abbreviation', add_special_tokens=False)[0]: 2,
            tokenizer.encode('Human', add_special_tokens=False)[0]: 3,
            tokenizer.encode('Numeric', add_special_tokens=False)[0]: 4,
            tokenizer.encode('Location', add_special_tokens=False)[0]: 5,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        if self.args.multiVerbalizer:
            multi_interest_index = []
            multi_interest_index.append(self.tokenizer.encode("Definition")[1])
            multi_interest_index.append(self.tokenizer.encode("Description")[1])
            multi_interest_index.append(self.tokenizer.encode("Manner")[1])
            multi_interest_index.append(self.tokenizer.encode("Reason")[1])
            multi_interest_index.append(self.tokenizer.encode("Animal")[1])
            multi_interest_index.append(self.tokenizer.encode("Body")[1])
            multi_interest_index.append(self.tokenizer.encode("Color")[1])
            multi_interest_index.append(self.tokenizer.encode("Creative")[1])
            multi_interest_index.append(self.tokenizer.encode("Currency")[1])
            multi_interest_index.append(self.tokenizer.encode("Diseases")[1])
            multi_interest_index.append(self.tokenizer.encode("Medicine")[1])
            multi_interest_index.append(self.tokenizer.encode("Event")[1])
            multi_interest_index.append(self.tokenizer.encode("Food")[1])
            multi_interest_index.append(self.tokenizer.encode("Instrument")[1])
            multi_interest_index.append(self.tokenizer.encode("Lang")[1])
            multi_interest_index.append(self.tokenizer.encode("Letter")[1])
            multi_interest_index.append(self.tokenizer.encode("Entity")[1])
            multi_interest_index.append(self.tokenizer.encode("Plant")[1])
            multi_interest_index.append(self.tokenizer.encode("Product")[1])
            multi_interest_index.append(self.tokenizer.encode("Religion")[1])
            multi_interest_index.append(self.tokenizer.encode("Sport")[1])
            multi_interest_index.append(self.tokenizer.encode("Substance")[1])
            multi_interest_index.append(self.tokenizer.encode("Symbol")[1])
            multi_interest_index.append(self.tokenizer.encode("Technique")[1])
            multi_interest_index.append(self.tokenizer.encode("Term")[1])
            multi_interest_index.append(self.tokenizer.encode("Vehicle")[1])
            multi_interest_index.append(self.tokenizer.encode("Word")[1])
            multi_interest_index.append(self.tokenizer.encode("Abbreviation")[1])
            multi_interest_index.append(self.tokenizer.encode("Expression")[1])
            multi_interest_index.append(self.tokenizer.encode("Group")[1])
            multi_interest_index.append(self.tokenizer.encode("Organization")[1])
            multi_interest_index.append(self.tokenizer.encode("Individual")[1])
            multi_interest_index.append(self.tokenizer.encode("Title")[1])
            multi_interest_index.append(self.tokenizer.encode("Person")[1])
            multi_interest_index.append(self.tokenizer.encode("Human")[1])
            multi_interest_index.append(self.tokenizer.encode("Code")[1])
            multi_interest_index.append(self.tokenizer.encode("Count")[1])
            multi_interest_index.append(self.tokenizer.encode("Date")[1])
            multi_interest_index.append(self.tokenizer.encode("Distance")[1])
            multi_interest_index.append(self.tokenizer.encode("Money")[1])
            multi_interest_index.append(self.tokenizer.encode("Order")[1])
            multi_interest_index.append(self.tokenizer.encode("Number")[1])
            multi_interest_index.append(self.tokenizer.encode("Period")[1])
            multi_interest_index.append(self.tokenizer.encode("Percent")[1])
            multi_interest_index.append(self.tokenizer.encode("Speed")[1])
            multi_interest_index.append(self.tokenizer.encode("Temperature")[1])
            multi_interest_index.append(self.tokenizer.encode("Size")[1])
            multi_interest_index.append(self.tokenizer.encode("Weight")[1])
            multi_interest_index.append(self.tokenizer.encode("Area")[1])
            multi_interest_index.append(self.tokenizer.encode("Volume")[1])
            multi_interest_index.append(self.tokenizer.encode("City")[1])
            multi_interest_index.append(self.tokenizer.encode("Country")[1])
            multi_interest_index.append(self.tokenizer.encode("Mountain")[1])
            multi_interest_index.append(self.tokenizer.encode("Location")[1])
            multi_interest_index.append(self.tokenizer.encode("State")[1])

            multi_logits = pred[:, multi_interest_index]
            c0 = multi_logits[:, :4].mean(dim=1)
            c1 = multi_logits[:, 4:27].mean(dim=1)
            c2 = multi_logits[:, 27:29].mean(dim=1)
            c3 = multi_logits[:, 29:35].mean(dim=1)
            c4 = multi_logits[:, 35:50].mean(dim=1)
            c5 = multi_logits[:, 50:].mean(dim=1)
            logits = torch.stack([c0, c1, c2, c3, c4, c5]).T

            label_map = self.label_map
            converted_target = target.clone()
            for key, val in label_map.items():
                converted_target[target == key] = val
            # interest_index = list(label_map.keys())
            # logits = logits[:, interest_index]

            # calculate hinge loss
            hinge_target = target.clone()
            for key, val in self.label_map.items():
                hinge_target[target==key] = val

            for t in hinge_target.cpu().numpy().tolist():
                self._target.append(t)
                
            pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            self.hinge += hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
            self._pred.extend(pred)
            self.ce_loss += self.ce_fct(pred, converted_target).item()
        else:
            # pred: batch_size x seq_len x vocab_size
            self.ce_loss += self.ce_fct(pred, target).item()

            # calculate hinge loss
            hinge_target = target.clone()
            for key, val in self.label_map.items():
                hinge_target[target==key] = val

            for t in hinge_target.cpu().numpy().tolist():
                self._target.append(t)

            interest_index = list(self.label_map.keys())
            pred = pred[:, interest_index]
            self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
            pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
            self._pred.extend(pred)


    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


# class DBPediaMetric(TaskMetric):
#     def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
#         super(DBPediaMetric, self).__init__(pred, target, seq_len, tokenizer)
#         self.label_map = {
#             tokenizer.encode("Company", add_special_tokens=False)[0]: 0,
#             tokenizer.encode("EducationalInstitution", add_special_tokens=False)[0]: 1,
#             tokenizer.encode("Artist", add_special_tokens=False)[0]: 2,
#             tokenizer.encode("Athlete", add_special_tokens=False)[0]: 3,
#             tokenizer.encode("OfficeHolder", add_special_tokens=False)[0]: 4,
#             tokenizer.encode("MeanOfTransportation", add_special_tokens=False)[0]: 5,
#             tokenizer.encode("Building", add_special_tokens=False)[0]: 6,
#             tokenizer.encode("NaturalPlace", add_special_tokens=False)[0]: 7,
#             tokenizer.encode("Village", add_special_tokens=False)[0]: 8,
#             tokenizer.encode("Animal", add_special_tokens=False)[0]: 9,
#             tokenizer.encode("Plant", add_special_tokens=False)[0]: 10,
#             tokenizer.encode("Album", add_special_tokens=False)[0]: 11,
#             tokenizer.encode("Film", add_special_tokens=False)[0]: 12,
#             tokenizer.encode("WrittenWork", add_special_tokens=False)[0]: 13,
#         }


class QNLIMetric(TaskMetric):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super(QNLIMetric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode('No', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 1,
        }


class QQPMetric(TaskMetric):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super(QQPMetric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode('No', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 1,
        }


class RTEMetric(MetricBase):
    def __init__(self, args=None, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('No', add_special_tokens=False)[0]: 1,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}
