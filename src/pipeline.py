"""
Closed-loop real-time feedback pipeline
"""

from pylsl import StreamInlet, resolve_stream
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

FS = 250
N_CHANNELS = 9
# 1 second to match the trained LDA
WINDOW_SECONDS = 1
WINDOW_SIZE = int(FS * WINDOW_SECONDS)
CHANNEL_IDX = [1, 3]

with open('model_weights.pkl', 'rb') as f:
    # the pickle contains the sk-learn object
    model = pickle.load(f)

streams = resolve_stream("type", "BCI_PREDICTION")
# or use this:
# streams = resolve_streams(1)
inlet = StreamInlet(streams[0])

window = SlidingWindow(size=WINDOW_SIZE, step=FS // 10, channels= 8)
PSD_extractor = BandPower(fs=FS, band=(8, 12))
logger = FeatureLogger(f"features_size_{WINDOW_SECONDS}.csv")

feedback = VisualBar()
controller = Controller()

# we want to take a window and:
# 1. apply bandpower filter
# 2. take 2 channels
# 3. bandpower
# 4. log10 
# 5. feed it to the LDA


while True:
    sample, _ = inlet.pull_sample()
    sample = sample[:8]
    win = window.update(sample)

    if win is not None:
        # extract the features to feed the LDA
        win = win[:, CHANNEL_IDX]
        psd = PSD_extractor.compute(win)
        feat = np.log10(psd)
        # logger.log(feat) 

        # feed the LDA
        pred = model.predict(feat)

        # update controller with prediction
        value = controller.update(pred[0]) # why [0]?
        feedback.update(value)


