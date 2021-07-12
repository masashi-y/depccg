from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax
from allennlp.training.metrics import CategoricalAccuracy
from depccg.allennlp.nn.bilinear import BilinearWithBias

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _apply_head_mask(attended_arcs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Mask the diagonal, because the head of a word can't be itself.
    attended_arcs = attended_arcs + \
        torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
    # Mask padded tokens, because we only want to consider actual words as heads.
    attended_arcs.masked_fill_((~mask).unsqueeze(1), -numpy.inf)
    attended_arcs.masked_fill_((~mask).unsqueeze(2), -numpy.inf)
    return attended_arcs


@Model.register("supertagger")
class Supertagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        tag_representation_dim: int,
        arc_representation_dim: int,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        dropout: float = 0.5,
        input_dropout: float = 0.5,
        head_tag_temperature: Optional[float] = None,
        head_temperature: Optional[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:
        super(Supertagger, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1,
            arc_representation_dim,
            Activation.by_name("elu")(),
            dropout=dropout
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim,
            arc_representation_dim,
            use_input_biases=True
        )

        num_labels = self.vocab.get_vocab_size("head_tags")

        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1,
            tag_representation_dim,
            Activation.by_name("elu")(),
            dropout=dropout
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearWithBias(
            tag_representation_dim,
            tag_representation_dim,
            num_labels
        )
        self._head_sentinel = torch.nn.Parameter(
            torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")
        self._input_dropout = Dropout(input_dropout)
        self._attachment_scores = CategoricalAccuracy()
        self._tagging_accuracy = CategoricalAccuracy()
        self.head_tag_temperature = head_tag_temperature
        self.head_temperature = head_temperature
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        words: Dict[str, torch.LongTensor],
        weight: torch.Tensor,
        metadata: List[Dict[str, Any]],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        embedded_text_input = self.text_field_embedder(words)
        embedded_text_input = self._input_dropout(embedded_text_input)

        mask = get_text_field_mask(words)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], 1)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self.head_arc_feedforward(encoded_text)
        child_arc_representation = self.child_arc_feedforward(encoded_text)

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self.head_tag_feedforward(encoded_text)
        child_tag_representation = self.child_tag_feedforward(encoded_text)
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(
            head_arc_representation,
            child_arc_representation
        )

        if head_indices is not None and head_tags is not None:
            loss, normalised_arc_logits, normalised_head_tag_logits = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                head_tags=head_tags,
                mask=mask,
                weight=weight
            )

            normalised_arc_logits = _apply_head_mask(
                normalised_arc_logits, mask)
            tag_mask = self._get_unknown_tag_mask(mask, head_tags)
            self._attachment_scores(
                normalised_arc_logits[:, 1:], head_indices[:, 1:], mask[:, 1:])
            self._tagging_accuracy(
                normalised_head_tag_logits[:, 1:], head_tags[:, 1:], tag_mask[:, 1:])
            predicted_heads, predicted_head_tags = None, None
        else:
            attended_arcs = _apply_head_mask(attended_arcs, mask)
            # Compute the heads greedily.
            # shape (batch_size, sequence_length)
            _, predicted_heads = attended_arcs.max(dim=2)

            # Given the greedily predicted heads, decode their dependency tags.
            # shape (batch_size, sequence_length, num_head_tags)
            head_tag_logits = self._get_head_tags(
                head_tag_representation,
                child_tag_representation,
                predicted_heads
            )
            _, predicted_head_tags = head_tag_logits.max(dim=2)

            loss, normalised_arc_logits, normalised_head_tag_logits = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                head_tags=predicted_head_tags.long(),
                mask=mask,
                weight=weight
            )
            normalised_arc_logits = _apply_head_mask(
                normalised_arc_logits, mask
            )

        output_dict = {
            "heads": normalised_arc_logits,
            "head_tags": normalised_head_tag_logits,
            "loss": loss,
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
        }

        if predicted_heads is not None and predicted_head_tags is not None:
            output_dict['predicted_heads'] = predicted_heads[:, 1:]
            output_dict['predicted_head_tags'] = predicted_head_tags[:, 1:]
            output_dict = self.decode(output_dict)
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        head_tags = output_dict.pop("head_tags")
        # discard a sentinel token and padding and unknown tags
        head_tags = head_tags[:, 1:, 2:]
        heads = output_dict.pop("heads")
        heads = heads[:, 1:]
        output_dict["head_tags"] = head_tags.cpu().detach().numpy()
        output_dict["heads"] = heads.cpu().detach().numpy()

        if 'predicted_heads' in output_dict:
            output_dict['predicted_heads'] = (
                output_dict['predicted_heads'].cpu().detach().numpy()
            )

        if 'predicted_head_tags' in output_dict:
            output_dict['predicted_head_tags'] = (
                output_dict['predicted_head_tags'].cpu().detach().numpy()
            )

        output_dict.pop('loss')
        return output_dict

    def _construct_loss(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        float_mask = mask.float()
        tag_mask = self._get_unknown_tag_mask(mask, head_tags)

        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(
            batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        if self.head_temperature:
            attended_arcs /= self.head_temperature
        normalised_arc_logits = (
            masked_log_softmax(attended_arcs, mask) *
            float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        )

        # shape (batch_size, sequence_length, num_head_tags)
        if self.head_tag_temperature:
            attended_arcs /= self.head_tag_temperature
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, head_indices
        )
        normalised_head_tag_logits = (
            masked_log_softmax(
                head_tag_logits, tag_mask.unsqueeze(-1)) * tag_mask.float().unsqueeze(-1)
        )
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(
            sequence_length, get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(
            batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[
            range_vector, child_index, head_indices
        ]
        tag_loss = normalised_head_tag_logits[
            range_vector, child_index, head_tags
        ]
        tag_loss *= (head_tags > 1).float()
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:] * weight
        tag_loss = tag_loss[:, 1:] * weight

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        loss = arc_nll + tag_nll
        return loss, normalised_arc_logits, normalised_head_tag_logits

    def _get_head_tags(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.Tensor
    ) -> torch.Tensor:

        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )

        return head_tag_logits

    def _get_unknown_tag_mask(
        self,
        mask: torch.LongTensor,
        head_tags: torch.LongTensor
    ) -> torch.LongTensor:
        oov = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'head_tags')
        new_mask = mask.detach()
        oov_mask = head_tags.eq(oov).long()
        new_mask = new_mask * (1 - oov_mask)
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        dependency = self._attachment_scores.get_metric(reset)
        tagging = self._tagging_accuracy.get_metric(reset)
        harmonic_mean = (2 * dependency * tagging) / (dependency + tagging)
        scores = {
            'dependency': dependency,
            'tagging': tagging,
            'harmonic_mean': harmonic_mean,
        }
        return scores
