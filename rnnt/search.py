import os

import utils
from tokenizer import SequenceTokenizer
from transcribe import export_all_transcriptions


def main():
    config = utils.load_config(os.environ.get("RNNT_CONFIG", "config/config.yaml"))
    logger = utils.setup_logger(config)
    device = utils.setup_device(logger)

    tokenizer = SequenceTokenizer(
        transcript_path=os.path.join(config.data.name, config.data.core_train),
        batch_size=config.training.batch_size,
        target_column=config.data.target_column or "transcript",
        mode=config.data.tokenizer or "char",
        target_normalization=config.data.target_normalization,
    )
    model = utils.initialize_model(config, tokenizer.vocab_size, device)
    export_all_transcriptions(model, config, tokenizer, device, logger)


if __name__ == "__main__":
    main()
