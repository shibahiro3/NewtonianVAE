class VisualHandlerBase:
    def __init__(self):
        self.title = ""

    def plot(self, *args, **kwargs) -> None:
        pass

    def wait_init(self):
        pass

    def call_end_init(self):
        pass

    def call_end(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return True
