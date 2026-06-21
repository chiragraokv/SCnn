import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class CIFARSCNN(nn.Module):
    def __init__(self,classes, num_steps=25,s_grad = surrogate.fast_sigmoid()):
        super().__init__()

        spike_grad = s_grad

        self.num_steps = num_steps
        self.classes = classes

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.lif5 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(256, self.classes)
        self.lif6 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        spk_out = []

        for t in range(self.num_steps):

            # If not using spike-encoded inputs,
            # the same image is presented every timestep

            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2 = self.pool(spk2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.bn4(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4 = self.pool(spk4)

            cur5 = self.fc1(spk4.flatten(1))
            spk5, mem5 = self.lif5(cur5, mem5)

            cur6 = self.fc2(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            spk_out.append(spk6)

        stacked_spikes = torch.stack(spk_out)
    
        return stacked_spikes.sum(dim=0)