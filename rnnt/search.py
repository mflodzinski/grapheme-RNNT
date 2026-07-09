import os

import utils
from tokenizer import GraphemeTokenizer
from transcribe import export_all_transcriptions


def main():
    config = utils.load_config(os.environ.get("RNNT_CONFIG", "config/config.yaml"))
    logger = utils.setup_logger(config)
    device = utils.setup_device(logger)

    tokenizer = GraphemeTokenizer(
        transcript_path=os.path.join(config.data.name, config.data.core_train),
        batch_size=config.training.batch_size,
    )
    model = utils.initialize_model(config, tokenizer.vocab_size, device)
    export_all_transcriptions(model, config, tokenizer, device, logger)


if __name__ == "__main__":
    main()
