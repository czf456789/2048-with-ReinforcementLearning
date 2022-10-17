import torch

import numpy as np
class Actor_Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_dim = 1
        action_dim = 4
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=256, padding=0, kernel_size=2, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(normalized_shape=[3, 3], eps=1e-05, elementwise_affine=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, padding=0, kernel_size=2, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(normalized_shape=[2, 2], eps=1e-05, elementwise_affine=True),

        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 * 2, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 4),
            torch.nn.LayerNorm(normalized_shape=[4], eps=1e-05, elementwise_affine=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        return torch.softmax(x, dim=1)


class Critic_Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_dim = 1
        action_dim = 4
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=256, padding=0, kernel_size=2, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(normalized_shape=[3, 3], eps=1e-05, elementwise_affine=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, padding=0, kernel_size=2, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(normalized_shape=[2, 2], eps=1e-05, elementwise_affine=True),

        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(512*2*2, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),

        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        return x

if __name__ == '__main__':
    a = np.array([[1, 20, 3, 4],
                  [5, 6, 7, 8, ],
                  [9, 10, 11, 12]], dtype=np.double)

    b = torch.from_numpy(a).type(torch.FloatTensor)
    c=torch.rand([1,2,2])

    print(c)
    x=torch.nn.LayerNorm([2,2],eps=1e-6)
    print(x(c))