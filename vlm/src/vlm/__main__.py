# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import logging
import argparse
from pathlib import Path
from gui.config import end_token
from colorama import Fore, Style
from vlm.modeling_vlm import make_VLM
from vlm.utils import setup_logging
from vlm.user_config import Config as user_params
from gui.generic_gui_interface import GenericGuiInterface
from chat_interface.config import ChatInterfaceConfig

BASE_DIR = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VLM options")
    logging_levels = list(logging._levelToName.values())

    parser.add_argument('-im',
                        '--input_image',
                        default=str(BASE_DIR / "test/data/industry.png"),
                        help="Input image path")
    parser.add_argument('-q',
                        '--question',
                        type=str,
                        help="Question, for test automation")
    parser.add_argument('-m',
                        '--model',
                        choices=['smolvlm-500M', 'smolvlm-256M'],
                        default='smolvlm-500M',
                        help="VLM to use.")
    parser.add_argument('-p',
                        '--precision',
                        choices=['q8', 'fp32'],
                        default='fp32',
                        help="Precision of the VLM")
    parser.add_argument('-l',
                        '--log_level',
                        type=str.upper,
                        choices=logging_levels,
                        default='INFO',
                        help="Verbose log level")
    parser.add_argument('-g', '--gui', dest='gui', action='store_true', help="Gui usage")
    parser.add_argument('-ng', '--no-gui', dest='gui', action='store_false', help="Gui usage")
    parser.set_defaults(gui=False)
    args = parser.parse_args()
    return args


def run_vision_async(vlm):
    vlm.image_features = vlm.run_vision(vlm.image_inputs)


if __name__ == '__main__':
    args = parse_args()
    setup_logging(level=logging._nameToLevel[args.log_level])

    print(f"{Fore.GREEN}{'Loading {} VLM {}...'.format(args.model, args.precision)}{Style.RESET_ALL}")

    vlm = make_VLM(args.model, args.precision, user_params=user_params, fixed_image=args.input_image)
    logger.info(f"{Fore.GREEN}{'Loaded! '}{Style.RESET_ALL}")

    # TODO: fix vision_threading slowdown process
    vlm.image_features = vlm.run_vision(vlm.image_inputs)
    if args.gui:
        gui = GenericGuiInterface(callback=None, user_config=ChatInterfaceConfig)
        gui.start()
        gui.send_connect()
        logger.info('Gui connected! ')

    print('Using {} image'.format(args.input_image))
    if args.gui:
        gui.send_cmd(args.input_image)
        gui.send_rsp(user_params.assistant_prompt)
        gui.send_rsp(end_token)

    question = args.question or input(user_params.assistant_prompt + "\n")
    logger.info(f"Question: {question}")

    prefilled = False  # Check if we have already computed vision for perf print

    while True:
        if args.gui:
            gui.send_qst(question)
        for i, decoded_token in enumerate(vlm.process_message(question)):
            if args.gui:
                gui.send_rsp(decoded_token)
            print(f"{Fore.LIGHTGREEN_EX}{decoded_token}{Style.RESET_ALL}", end="")

        if args.gui:
            gui.send_rsp(end_token)
            if prefilled:
                vlm.perf['vision'] = 0.0
            gui.send_cmd(vlm.str_perf())
        print(f"{Fore.LIGHTGREEN_EX}\n{vlm.str_perf()}{Style.RESET_ALL}")
        prefilled = True
        question = input("\n>")
