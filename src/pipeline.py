"""
Closed-loop real-time feedback pipeline
"""

from pylsl import StreamInlet, resolve_streams
from src.feedback import VisualBar
from src.controller import Controller
from src.utils.bandpower import BandPower
from src.utils.window import SlidingWindow
from src.utils.log import FeatureLogger
import numpy as np
import pickle
# for the loaded model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter
from scipy.signal import butter


FS = 250
N_CHANNELS = 9
# 1 second to match the trained LDA
WINDOW_SECONDS = 1
WINDOW_SIZE = int(FS * WINDOW_SECONDS)
CHANNEL_IDX = [1, 3]
BAND = [8, 30]

with open('model_weights.pkl', 'rb') as f:
    # the pickle contains the sk-learn object
    model = pickle.load(f)

streams = resolve_streams(1)
# or use this:
# streams = resolve_streams(1)
inlet = StreamInlet(streams[0])

window = SlidingWindow(size=WINDOW_SIZE, step=FS // 10, channels= 8)
# logger = FeatureLogger(f"features_size_{WINDOW_SECONDS}.csv")

feedback = VisualBar()
controller = Controller(smoothing=0.05)

# we want to take a window and:
# 1. take 2 channels
# 2. lfilter (because we need a causal filter)
# 3. log10 
# 4. feed it to the LDA


b, a =  butter(4, BAND, fs = FS, btype= 'band')

window.start_clock()

while True:
    sample, _ = inlet.pull_sample()
    sample = sample[:8]
    win = window.update(sample)

    if win is not None:
        # extract the features to feed the LDA
        win = win[:, CHANNEL_IDX]
        filtered_win = lfilter(b, a, win)
        psd = np.square(filtered_win)
        psd = psd.mean(axis = 0)
        feat = np.log10(psd)
        feat = np.expand_dims(feat, axis = 0)
        # logger.log(feat) 

        # feed the LDA
        pred = model.predict(feat)

        # update controller with prediction
        print(f"the prediction is {pred}")
        if pred[0] == 'IMAGERY_LEFT':
            value = controller.update(0)
        else:
            value = controller.update(1)
        feedback.update(value)


