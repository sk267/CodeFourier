import random


class UNet_Label_Gen_Fake():
    def __init__(self):
        self.counter = 1.0

    def get_random_alias_value(self):
        return random.uniform(0, 1)

    def get_decreasing_alias_value(self):
        self.counter -= 0.05
        return self.counter
