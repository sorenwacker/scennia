from timeit import default_timer as timer

import dash


class Timer:
    def __init__(self, name: str, description: str | None = None):
        self.name = name
        self.description = description
        self.start_time = timer()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.record()

    def start(self):
        self.start_time = timer()

    def record(self):
        dash.callback_context.record_timing(self.name, timer() - self.start_time, self.description)
