# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: 'Python 3.6.9 64-bit (''visord'': venv)'
#     name: python36964bitvisordvenvfc106c38ce1c4db7ab15237db0033b94
# ---

# # Object tracking tests

# ## Imports

import cv2
import numpy as np
import utils

frames = utils.read_video("./data/juggling_front.mp4")
utils.display_frames(frames)


