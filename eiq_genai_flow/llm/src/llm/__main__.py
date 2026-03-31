# Copyright 2024-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import argparse
import time
from llm.modeling_llm import make_LLM
from llm.config.user_config import Config as user_config
from llm.config.models_config import get_model_names
from llm.utils import print_models
import logging
import os
from shared_utils.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM options")
    logging_levels = list(logging._levelToName.values())

    parser.add_argument('-m', '--model', choices=get_model_names(), default="danube-500M-q8", help="LLM to use.")
    parser.add_argument('-p', '--prompt', default="You are an NXP assistant. Answer to the user query.")
    parser.add_argument('-l',
                        '--log_level',
                        type=str.upper,
                        choices=logging_levels,
                        default='INFO',
                        help="Verbose log level")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    logger.info('Available LLM(s):')
    print_models(get_model_names())
    args = parse_args()

    setup_logging(
        level=logging._nameToLevel[args.log_level],
        root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    llm = make_LLM(name=args.model, user_params=user_config)
    if llm.user_params.timeout is not None:
        llm.timeout.cancel()
    logger.info(f"Using {llm.name} model")
    while True:
        input_question = input('\nType your question here: ')
        logger.info(input_question)
        lower_question = input_question.lower()
        start_t = time.time()
        for i, decoded_token in enumerate(llm(lower_question, user_config.prompt)):
            if i == 0:
                ttft = time.time() - start_t
                start_t = time.time()
            print(decoded_token, end='')
        logger.info(f"\nTTFT:{ttft}")
