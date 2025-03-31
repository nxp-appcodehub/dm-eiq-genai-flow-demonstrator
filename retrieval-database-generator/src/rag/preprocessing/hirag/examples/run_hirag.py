# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from rag.utils import pretty_print
from hirag.hirag_text_splitters import HiRAGTextSplitter
from hirag.config import Config as HiRAGConfig


if __name__ == "__main__":
    config = HiRAGConfig(global_understanding_QA_pair_limit=5,
                         local_QA_pair_limit=5,
                         data_augmentation_QA_pair_limit=2)
    output = {}

    context = """
        To purchase and download games, you'll need to connect your console to the internet and sign in to your account.
        PlayStation Network and PlayStation Store subject to terms of use and country and language restrictions. Users are responsible for internet service fees. Charges apply for some content and / or services. Users must be 7 years or older and users under 18 require parental consent. Additional age restrictions may apply. Service availability is not guaranteed. Online features of specific games may be withdrawn on reasonable notice - playstation.com/gameservers. Full terms apply: PSN Terms of Service at playstation.com/legal.
        Each time you press the mute button, your mic switches between muted (button lit) and unmuted (button off). Press and hold the mute button to mute your mic and to turn off sound output from the speakers on your controller and TV. Press the mute button again to return to the original state.
        You can use up to 4 controllers at once. Press the (PS) button to assign numbers to your controllers. The player indicator lights turn on accordingly. Numbers are assigned in order from 1, and you can determine your controller's number by the number of lights that turn on.
        Your PS5 console's power saving mode is called rest mode. You can do things like charge your controller via the console's USB ports, automatically update your system software, and keep your game or app suspended while powered down. To find out which rest mode settings are optimal for you, see the User's Guide (page 13).
    """
    
    text_splitter = HiRAGTextSplitter(config=config)

    chunks = text_splitter.generate_hirag_chunks(data=context)

    pretty_print(name="HiRAG example using dummy text about PlayStation", result_dictionary=chunks)
