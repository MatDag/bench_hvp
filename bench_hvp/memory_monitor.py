from threading import Thread

from pynvml import nvmlDeviceGetMemoryInfo
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex


class GPUMemoryMonitor(Thread):
    """Monitor the memory usage in MB in a separate thread.

    Note that this class is good enough to highlight the memory profile of
    Parallel in this example, but is not a general purpose profiler fit for
    all cases.
    """
    def __init__(self):
        super().__init__()
        self.stop = False
        self.memory_buffer = []
        self.start()

    def run(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        memory_start = nvmlDeviceGetMemoryInfo(handle).used
        while not self.stop:
            self.memory_buffer.append(nvmlDeviceGetMemoryInfo(handle).used
                                      - memory_start)

    def join(self):
        self.stop = True
        super().join()
