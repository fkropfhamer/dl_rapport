class Layer:
    def forward(self, x):
        self.last_input = x

        return self._forward(x)

    def __call__(self, x):
        return self.forward(x)