from pytorch_pretrained_bert import BertModel
from pytorch_transformers import RobertaModel
from transformers import GPT2Model, GPT2LMHeadModel, PretrainedConfig, GPT2Config
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch import nn, cat
import torch.nn.functional as F


class BertModelForStanceClassification(BertPreTrainedModel):
    """
        `text`: a torch.LongTensor of shape [batch_siz, sequence_length]
        `type_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).

        `input_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
    """

    def __init__(self, config, classes=4):
        super(BertModelForStanceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.last_layer = nn.Linear(config.hidden_size, classes)
        self.apply(self.init_bert_weights)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def reinit(self, config):
        self.dropout = nn.Dropout(config["hyperparameters"]["hidden_dropout_prob"])

    def forward(self, batch):
        _, pooled_output = self.bert(batch.text, batch.type_mask, batch.input_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.last_layer(pooled_output)
        return logits

class RoBertaModelForStanceClassification(RobertaModel):
    """
        `text`: a torch.LongTensor of shape [batch_siz, sequence_length]
    """

    def __init__(self, config, classes=4):
        super(RoBertaModelForStanceClassification, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.last_layer = nn.Linear(config.hidden_size, classes)
        # self.apply(self.init_weights)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def reinit(self, config):
        self.dropout = nn.Dropout(config["hyperparameters"]["hidden_dropout_prob"])

    def forward(self, batch):
        _, pooled_output = self.roberta(batch.text)
        pooled_output = self.dropout(pooled_output)
        logits = self.last_layer(pooled_output)
        return logits

class RoBertaWFeaturesModelForStanceClassification(RobertaModel):
    """
        `text`: a torch.LongTensor of shape [batch_siz, sequence_length]
    """

    def __init__(self, config, classes=4):
        super(RoBertaWFeaturesModelForStanceClassification, self).__init__(config)
        self.roberta = RobertaModel(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_layer = nn.Linear(config.hidden_size + 9, config.hidden_size)
        
        self.last_layer = nn.Linear(config.hidden_size, classes)

    def reinit(self, config):
        self.dropout = nn.Dropout(config["hyperparameters"]["hidden_dropout_prob"])

    def forward(self, batch):
        _, pooled_output = self.roberta(batch.text)

        used_features = [
            batch.hasqmark,
            batch.hasemark,
            batch.hashashtag,
            batch.hasurl,
            batch.haspic,
            batch.hasnegation,
            batch.hasswearwords,
            batch.src_rumour,
            batch.thread_rumour#,
            # batch.spacy_processed_NERvec
        ]
        features = cat(tuple([f.unsqueeze(-1) for f in used_features]), dim=-1)

        pooled_output = self.dropout(cat((pooled_output, features), 1))

        outp_with_features = F.relu(self.dropout(self.hidden_layer(pooled_output)))

        logits = self.last_layer(outp_with_features)
        return logits

class GPT2ModelForStanceClassification(GPT2Model):

    def __init__(self, config, classes=4):
        super(GPT2ModelForStanceClassification, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.last_layer = nn.Linear(config.hidden_size, classes)
        # self.apply(self.init_weights)
        self.dropout = nn.Dropout(config.output_hidden_states)

    def reinit(self, config):
        self.dropout = nn.Dropout(config["hyperparameters"]["hidden_dropout_prob"])

    def forward(self, batch):
        gpt_out, pooled_output = self.gpt2(batch.text).logits
        batch_size = gpt_out.shape[0]
        # pooled_output = self.dropout(pooled_output.last_hidden_state[0])
        logits = self.last_layer(gpt_out.view(batch_size,-1))
        return logits

    # def from_pretrained(self, config):
    #     self.gpt2 = GPT2LMHeadModel.from_pretrained(config)