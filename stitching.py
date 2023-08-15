class Stitching:
    def __init__(self, method, background_temperature):
        self.method: str = method
        self.background_temperature: float = background_temperature

    def __call__(self, current_value: float, additional_value: float):
        if self.method == "max":
            return max(current_value, additional_value)
        elif self.method == "add":
            if current_value == self.background_temperature:
                return additional_value
            else:
                return current_value + additional_value - self.background_temperature

