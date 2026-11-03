"""🖼️ Batch augmentations"""

from torchvision.transforms.v2 import Transform


class StandardizeEachSample(Transform):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def _transform(self, inpt, params):
        mean = inpt.mean(dim=[2, 3], keepdim=True)
        std = inpt.std(dim=[2, 3], keepdim=True)
        return (inpt - mean) / (std + self.eps)

    def __call__(self, inpt):
        return self._transform(inpt, None)
