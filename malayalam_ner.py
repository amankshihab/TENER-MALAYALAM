import torch

from bpemb import BPEmb

import os
from collections import OrderedDict
import string
import unicodedata


class MalayalamNER:
    def __init__(self, model_name: str, path_to_weights: str, device: str='cpu'):
        if model_name == None:
            raise AttributeError("model_name is a required attribute")
        if path_to_weights == None:
            raise AttributeError("path_to_weights is a required attribute")
        self.model_name = model_name.lower()
        assert self.model_name in ('bilstm', 'tener'), "Model not supported"
        self.path_to_weights = path_to_weights
        self.device = device
        self.model = self._load_model().to(device)
        self.bpemb = BPEmb(lang='ml', add_pad_emb=True)

    def _load_model(self) -> torch.nn.Module:
        if self.model_name == 'bilstm':
            from models.bilstm import bi_lstm
            _model = bi_lstm
        elif self.model_name == 'tener':
            from models.tener_ml import tener_ml
            _model = tener_ml
        else:
            raise NotImplemented("Requested model is not supported")
        path = os.path.join(self.path_to_weights, self.model_name + '.ckpt')
        state_dict = torch.load(path, map_location=torch.device(self.device))['state_dict']
        new_state_dict = OrderedDict((key.replace('model.', ''), value) for key, value in state_dict.items())
        _model.load_state_dict(new_state_dict)
        return _model
    
    def _get_state_dict(self) -> OrderedDict:
        try:
            path = os.path.join(self.path_to_weights, self.model_name + '.ckpt')
            print(path)
            state_dict = torch.load(path, map_location=torch.device(self.device))['state_dict']
            new_state_dict = OrderedDict((key.replace('model.', ''), value) for key, value in state_dict.items())
            return new_state_dict
        except FileNotFoundError:
            print("Failed to find file containing state_dict of the model.")
        except Exception as e:
            print(e)

    def _remove_punctuation(self, input_string: str) -> str:
        return input_string.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_unicode_control_characters(self, input_string: str) -> str:
        return ''.join(char for char in input_string if not unicodedata.category(char).startswith('C'))
    
    def _inference_preprocess(self, input_string: str) -> str:
        input_string = self._remove_punctuation(input_string)
        input_string = self._remove_unicode_control_characters(input_string)
        return " ".join(input_string.split())

    def _get_tokens_and_masks(self, input_string):
        tokens = self.bpemb.encode_ids_with_bos_eos(input_string)
        padding_len = 100 - len(tokens)
        if padding_len < 0:
            tokens = tokens[:99]
            tokens = tokens + [self.bpemb.EOS]
            tokens = torch.tensor(tokens, dtype=torch.int64)
        else:
            padding_vectors = torch.zeros(padding_len) - 1
            tokens = torch.tensor(tokens, dtype=torch.int64)
            tokens = torch.cat([tokens, padding_vectors], axis=0)
        mask = tokens > 0
        tokens[~mask] = 0
        return tokens.long().unsqueeze(0), mask.long().unsqueeze(0)

    def predict(self, input_string: str) -> list[tuple[str, str, float]]:
        ids_to_tags = {
            0 : 'O',
            1 : 'B-PER',
            2 : 'I-PER',
            3 : 'B-ORG',
            4 : 'I-ORG',
            5 : 'B-LOC',
            6 : 'I-LOC'
        }
        input_string = self._inference_preprocess(input_string)
        tokens, mask = self._get_tokens_and_masks(input_string)
        tokens = tokens.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():
            if self.model_name == 'bilstm':
                op = self.model(tokens)
            elif self.model_name == 'tener':
                op = self.model(tokens, mask)
        softmaxed = torch.softmax(op, dim=-1)[mask == 1]
        conf_scores = torch.max(softmaxed, dim=-1)
        conf_scores = list(map(lambda x: x.item(), conf_scores[0]))
        argmaxed = torch.argmax(op, dim=-1)[mask == 1]
        # tags = [ids_to_tags[i.item()] for i in argmaxed]
        # return list(zip(self.bpemb.encode_with_bos_eos(input_string), tags, conf_scores))
        return list(zip(self.bpemb.encode_with_bos_eos(input_string), argmaxed, conf_scores))