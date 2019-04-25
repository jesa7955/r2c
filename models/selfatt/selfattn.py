"""
Attention is all you need!
"""

from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, \
                             InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, \
                             add_positional_features, replace_masked_values, \
                             add_sentence_boundary_token_ids
from allennlp.nn import InitializerApplicator

@Model.register("MultiModalAttentionQA")
class MultiModalAttentionQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 fusion_encoder: Seq2SeqEncoder,
                 type_vocab_size: int = 3,
                 feature_dim: int = 768,
                 final_mlp_hidden_dim: int = 1024,
                 input_dropout: float = 0.3,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(MultiModalAttentionQA, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True,
                                       average_pool=True,
                                       semantic=class_embs,
                                       final_dim=feature_dim)
        ######################################################################

        self.token_type_embeddings = nn.Embedding(type_vocab_size, feature_dim)
        self.bos_token = torch.randn(feature_dim)
        self.eos_token = torch.randn(feature_dim)

        self.encoder_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.feature_dim = feature_dim
        self.fusion_encoder = TimeDistributed(fusion_encoder)

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question

        final_mlp_dim = fusion_encoder.get_output_dim()
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(final_mlp_dim, final_mlp_hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(final_mlp_hidden_dim, 1),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        ##################################################
        # Concatenate words features and object features #
        # at the dim of sequence                         #
        ##################################################

        obj_features = obj_reps['obj_reps']
        obj_bs, obj_len, obj_dim = obj_features.shape
        que_bs, a_num, que_len, que_dim = question['bert'].shape
        ans_bs, a_num, ans_len, ans_dim = answers['bert'].shape

        # Add [SEP] and [CLS]. What is really done here is wrap question,
        # answers, and images obejcts with <S> </S> then remove the last
        # two <S> and view the first one as [CLS]
        question_bert, question_mask = add_sentence_boundary_token_ids(
                                           question['bert'].view(-1, que_len, que_dim),
                                           question_mask,
                                           self.bos_token.to(question_mask.device),
                                           self.eos_token.to(question_mask.device))
        question_bert = question_bert.view(que_bs, a_num, que_len+2, que_dim)
        question_mask = question_mask.view(que_bs, a_num, que_len+2)
        answers_bert, answer_mask = add_sentence_boundary_token_ids(
                                        answers['bert'].view(-1, ans_len, ans_dim),
                                        answer_mask,
                                        self.bos_token.to(answer_mask.device),
                                        self.eos_token.to(answer_mask.device))
        answers_bert = answers_bert.view(ans_bs, a_num, ans_len+2, ans_dim)[:, :, 1:, :]
        answer_mask = answer_mask.view(ans_bs, a_num, ans_len+2)[:, :, 1:]
        obj_features, obj_mask = add_sentence_boundary_token_ids(
                                     obj_features,
                                     box_mask,
                                     self.bos_token.to(box_mask.device),
                                     self.eos_token.to(box_mask.device))
        obj_features = obj_features.view(obj_bs, obj_len+2, obj_dim)[:, 1:, :]
        obj_mask = obj_mask.view(obj_bs, obj_len+2)[:, 1:]
        obj_features = torch.stack([obj_features for _ in range(a_num)], dim=1)
        obj_mask = torch.stack([obj_mask for _ in range(a_num)], dim=1)
        # The shape for the input of transformer is
        # batch_size * num_answers * new_seq_length * dim
        # where new_seq_length = question_seq_length + 2 +
        #                        answer_seq_lenght + 1 +
        #                        max_num_objects + 1
        que_ans_obj = torch.cat((question_bert,
                                  answers_bert,
                                  obj_features), dim=2)
        que_ans_obj_mask = torch.cat((question_mask,
                                      answer_mask,
                                      obj_mask), dim=2)

        # Add positional features
        total_bs, a_num, total_len, total_dim = que_ans_obj.shape
        que_ans_obj = add_positional_features(que_ans_obj.view(-1,
                                                               total_len,
                                                               total_dim)).view(total_bs,
                                                                                a_num,
                                                                                total_len,
                                                                                total_dim)

        # Add type information, which is used to distinguished between
        # Qution, Answer, and Images
        target_device = que_ans_obj.device
        question_type_ids = torch.zeros(que_bs, a_num, que_len+2, dtype=torch.long, device=target_device)
        answers_type_ids = 1 - torch.zeros(ans_bs, a_num, ans_len+1, dtype=torch.long, device=target_device)
        objs_type_ids = 2 - torch.zeros(obj_bs, a_num, obj_len+1, dtype=torch.long, device=target_device)
        token_type_ids = torch.cat((question_type_ids,
                                    answers_type_ids,
                                    objs_type_ids), dim=2)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        que_ans_obj = que_ans_obj + token_type_embeddings

        ##########################################
        # Self attetion
        outputs = self.fusion_encoder(que_ans_obj, que_ans_obj_mask)
        bs, a_num, seq_len, output_dim = outputs.shape
        cls_reps = outputs[:, :, 1, :].squeeze(2)

        ###########################################

        logits = self.final_mlp(cls_reps.view(-1, output_dim)).view(bs, a_num)

        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
