class ModelProvider:
    def __init__(self, models: dict[str, object]) -> None:
        self._models = models

    def get(self, name: str):
        return self._models[name]
