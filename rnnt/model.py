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
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0,
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


def build_encoder(config, vocab_size):
    if config.enc.type == "lstm":
        return Encoder(
            input_size=config.num_features,
            hidden_size=config.enc.hidden_size,
            output_size=vocab_size,
            num_layers=config.enc.num_layers,
            dropout=config.enc.dropout,
            bidirectional=config.enc.bidirectional,
        )
    raise NotImplementedError


def build_decoder(config, vocab_size):
    if config.dec.type == "lstm":
        return Decoder(
            hidden_size=config.dec.hidden_size,
            vocab_size=vocab_size,
            embedding_size=config.dec.embedding_size or config.dec.hidden_size,
            output_size=vocab_size,
            num_layers=config.dec.num_layers,
            dropout=config.dec.dropout,
        )
    raise NotImplementedError


class JointNet(nn.Module):
    def __init__(self):
        super(JointNet, self).__init__()

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        return enc_state + dec_state


class Transducer(nn.Module):
    def __init__(self, config, vocab_size, device):
        super(Transducer, self).__init__()
        self.config = config
        self.device = device
        self.encoder = build_encoder(config, vocab_size)
        self.decoder = build_decoder(config, vocab_size)

        self.joint = JointNet()
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
    def recognize(self, inputs, input_lengths):
        zero_token = torch.tensor([self.blank], device=self.device)
        batch_size = inputs.shape[0]

        encoded_sequences, _ = self.encoder(inputs, input_lengths)
        decoded_sequences = [
            self.decode_sequence(
                encoded_sequences[i], input_lengths[i], zero_token
            )
            for i in range(batch_size)
        ]

        return decoded_sequences

    @torch.no_grad()
    def decode_sequence(self, encoded_sequence, input_length, zero_token):
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
