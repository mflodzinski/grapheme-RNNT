import heapq
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNTLossAdapter(nn.Module):
    def __init__(self, blank):
        super(RNNTLossAdapter, self).__init__()
        self.blank = blank
        try:
            from warprnnt_pytorch import RNNTLoss
        except ImportError:
            try:
                from torchaudio.transforms import RNNTLoss
            except ImportError as exc:
                raise ImportError(
                    "RNN-T training requires an RNN-T loss implementation. "
                    "Install torchaudio or warprnnt_pytorch."
                ) from exc

        self.loss = RNNTLoss(blank=blank, reduction="none")

    def forward(self, logits, targets, inputs_length, targets_length):
        return self.loss(logits, targets.contiguous(), inputs_length, targets_length)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        bidirectional=True,
    ):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.proj = EncoderProjection(hidden_size, output_size)

    def forward(self, inputs, input_lengths):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, sorted_seq_lengths.cpu(), batch_first=True
            )

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.proj(outputs)
        return logits, hidden


class EncoderProjection(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EncoderProjection, self).__init__()
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        forward_output, backward_output = x.chunk(2, dim=-1)
        forward_projected = self.linear1(forward_output)
        backward_projected = self.linear2(backward_output)
        return forward_projected + backward_projected


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        embedding_size,
        output_size,
        num_layers,
        dropout,
        padding_idx=None,
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs.long())

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths.cpu(), batch_first=True
            )

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.proj(outputs)
        return logits, hidden


def build_encoder(config, output_size):
    if config.enc.type == "lstm":
        return Encoder(
            input_size=config.num_features,
            hidden_size=config.enc.hidden_size,
            output_size=output_size,
            num_layers=config.enc.num_layers,
            dropout=config.enc.dropout,
            bidirectional=config.enc.bidirectional,
        )
    raise NotImplementedError


def build_decoder(config, vocab_size, output_size, padding_idx=None):
    if config.dec.type == "lstm":
        return Decoder(
            hidden_size=config.dec.hidden_size,
            vocab_size=vocab_size,
            embedding_size=config.dec.embedding_size or config.dec.hidden_size,
            output_size=output_size,
            num_layers=config.dec.num_layers,
            dropout=config.dec.dropout,
            padding_idx=padding_idx,
        )
    raise NotImplementedError


class AdditiveJointNet(nn.Module):
    def __init__(self, vocab_size):
        super(AdditiveJointNet, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)
        else:
            assert enc_state.dim() == dec_state.dim()

        return enc_state + dec_state


def logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    largest = max(a, b)
    return largest + math.log(math.exp(a - largest) + math.exp(b - largest))


def prune_hypotheses(hypotheses, beam_width):
    merged = {}
    for hyp in hypotheses:
        tokens = hyp["tokens"]
        existing = merged.get(tokens)
        if existing is None:
            merged[tokens] = hyp
            continue
        existing["score"] = logaddexp(existing["score"], hyp["score"])

    return sorted(
        merged.values(),
        key=lambda hyp: hyp["score"],
        reverse=True,
    )[:beam_width]


class Transducer(nn.Module):
    def __init__(self, config, vocab_size, device, pad_idx=None):
        super(Transducer, self).__init__()
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        joint_size = vocab_size
        self.encoder = build_encoder(config, joint_size)
        self.decoder = build_decoder(
            config,
            vocab_size,
            joint_size,
            padding_idx=pad_idx,
        )

        self.joint = AdditiveJointNet(vocab_size=vocab_size)
        self.blank = vocab_size - 1
        self.crit = RNNTLossAdapter(blank=self.blank)

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state, _ = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=self.blank)
        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)
        loss = self.crit(logits, targets.contiguous(), inputs_length, targets_length)
        return loss

    @torch.no_grad()
    def recognize(self, inputs, input_lengths, decode_method=None):
        batch_size = inputs.shape[0]
        decode_config = self.config.decode
        method = (
            "greedy"
            if decode_config is None or decode_config.method is None
            else decode_config.method
        )
        if decode_method is not None:
            method = decode_method

        encoded_sequences, _ = self.encoder(inputs, input_lengths)
        decoded_sequences = []
        for i in range(batch_size):
            encoded_sequence = encoded_sequences[i]
            input_length = int(input_lengths[i].item())
            if method == "greedy":
                decoded_sequences.append(
                    self.decode_sequence_greedy(encoded_sequence, input_length)
                )
            elif method == "beam":
                decoded_sequences.append(
                    self.decode_sequence_beam(encoded_sequence, input_length)
                )
            else:
                raise ValueError(f"Unsupported decode method: {method}")

        return decoded_sequences

    @torch.no_grad()
    def decode_sequence_greedy(self, encoded_sequence, input_length):
        zero_token = torch.tensor([self.blank], device=self.device)
        preds = []
        u = 0
        t = 0
        u_max = self.config.max_length
        gu, hidden = self.decoder(zero_token)

        while t < input_length and u < u_max:
            h = self.joint(encoded_sequence[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            _, pred = torch.max(out, dim=0)
            pred = int(pred.item())

            if pred != self.blank:
                preds.append(pred)
                token = torch.tensor([pred], device=self.device)
                gu, hidden = self.decoder(token, hidden=hidden)
                u += 1
            else:
                t += 1

        return preds

    @torch.no_grad()
    def decode_sequence_beam(self, encoded_sequence, input_length):
        decode_config = self.config.decode
        beam_width = max(1, decode_config.beam_width or 5)
        label_tokens = [
            token
            for token in range(self.vocab_size)
            if token != self.blank and token != self.pad_idx
        ]
        zero_token = torch.tensor([self.blank], device=self.device)
        initial_dec_state, initial_hidden = self.decoder(zero_token)
        beam = [
            {
                "tokens": tuple(),
                "score": 0.0,
                "dec_state": initial_dec_state.view(-1),
                "hidden": initial_hidden,
            }
        ]

        for t in range(input_length):
            completed = []
            active = []
            active_counter = 0
            for hyp in beam:
                heapq.heappush(active, (-hyp["score"], active_counter, hyp))
                active_counter += 1

            while active:
                best_active_score = -active[0][0]
                stronger_completed = sum(
                    hyp["score"] > best_active_score for hyp in completed
                )
                if stronger_completed >= beam_width:
                    break

                _, _, hyp = heapq.heappop(active)
                logits = self.joint(
                    encoded_sequence[t].view(-1),
                    hyp["dec_state"].view(-1),
                )
                log_probs = F.log_softmax(logits, dim=0)

                completed.append(
                    {
                        "tokens": hyp["tokens"],
                        "score": hyp["score"] + float(log_probs[self.blank].item()),
                        "dec_state": hyp["dec_state"],
                        "hidden": hyp["hidden"],
                    }
                )

                if len(hyp["tokens"]) >= self.config.max_length:
                    continue

                for token in label_tokens:
                    token_tensor = torch.tensor([token], device=self.device)
                    dec_state, hidden = self.decoder(
                        token_tensor,
                        hidden=hyp["hidden"],
                    )
                    heapq.heappush(
                        active,
                        (
                            -hyp["score"] - float(log_probs[token].item()),
                            active_counter,
                            {
                                "tokens": hyp["tokens"] + (token,),
                                "score": hyp["score"] + float(log_probs[token].item()),
                                "dec_state": dec_state.view(-1),
                                "hidden": hidden,
                            },
                        ),
                    )
                    active_counter += 1

            if completed:
                beam = prune_hypotheses(completed, beam_width)
            else:
                beam = prune_hypotheses(
                    [heap_item[2] for heap_item in active],
                    beam_width,
                )

        best_hypothesis = max(
            beam,
            key=lambda hyp: hyp["score"] / max(1, len(hyp["tokens"])),
        )
        return list(best_hypothesis["tokens"])
