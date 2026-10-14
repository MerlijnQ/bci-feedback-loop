"""
Sliding window for streaming EEG samples; Code adjusted to the implementation of the previous assignment.
"""

import numpy as np
import pylsl as lsl

class SlidingWindow:
    def __init__(self, size, step, channels):
        self.size = size
        self.step = step
        self.channels = channels
        self.buffer = np.empty((0, channels))
        self.counter = 0
        self.is_full = False
        self.start_time = None

    # clock to measure how long to fill up the first window
    def start_clock(self):
        self.start_time = lsl.local_clock()

    def update(self, sample):
            
        sample = np.array(sample)
        self.buffer = np.vstack((self.buffer, sample))

        if self.is_full:
            self.counter += len(sample)

        # sliding window: keep the last 'window size' samples
        if len(self.buffer) > self.size:
            self.buffer = self.buffer[-self.size:]      

        # return window only when the 'sliding' reaches the step size 
        if len(self.buffer) == self.size and self.counter % self.step == 0:

            if not self.is_full:
                self.is_full = True
                end_time = lsl.local_clock()
                time_span = end_time - self.start_time
                print(f"Time taken to full up buffer: {time_span:.2f} seconds")
                print(f"Size of window {self.buffer.shape}")

            
            if len(self.buffer.shape) > 1:
                # removed the transpose to keep the shape as (samples, channels) for the Welch
                return self.buffer
            else:
                return self.buffer
            

        return None

