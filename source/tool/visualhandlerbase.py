class VisualHandlerBase:
    def __init__(self):
        pass

    def plot(self, *args, **kwargs) -> None:
        pass

    def call_end(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return True
