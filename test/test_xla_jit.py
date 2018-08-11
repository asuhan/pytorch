import torch
import torch.nn as nn
import torch.nn.functional as F
from common import TestCase, run_tests


def _xla_run(model, input):
    traced_model = torch.jit.trace(input)(model)
    return torch._C._to_xla_module(traced_model)(input)


class TestMulAdd(TestCase):
    def test(self):

        class XlaMulAdd(nn.Module):
            def forward(self, x, y):
                return x * y + y

        x = torch.rand(3, 5)
        y = torch.rand(3, 5)
        model = XlaMulAdd()
        traced_model = torch.jit.trace(x, y)(model)
        out = torch._C._to_xla_module(traced_model)(x, y)
        expected = model(x, y)
        self.assertEqual(out.data, expected.data)


class TestRelu(TestCase):
    def test(self):

        class XlaRelu(nn.Module):
            def forward(self, x):
                return F.relu(x)

        x = torch.randn(2, 1, 4, 6)
        model = XlaRelu()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestThreshold(TestCase):
    def test(self):

        class XlaThreshold(nn.Module):
            def __init__(self):
                super(XlaThreshold, self).__init__()
                self.threshold = nn.Threshold(0.4, 20)

            def forward(self, x):
                return self.threshold(x)

        x = torch.rand(4, 2)
        model = XlaThreshold()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestTranspose(TestCase):
    def test(self):

        class XlaTranspose(nn.Module):
            def forward(self, x):
                return torch.t(x)

        x = torch.rand(2, 3)
        model = XlaTranspose()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestView(TestCase):
    def test(self):

        class XlaView(nn.Module):
            def forward(self, x):
                return x.view(-1, 16)

        x = torch.rand(4, 8)
        model = XlaView()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestExpand(TestCase):
    def test(self):

        class XlaExpand(nn.Module):
            def forward(self, x):
                return x.expand(2, 5)

        x = torch.rand(5)
        model = XlaExpand()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestLinear(TestCase):
    def test(self):

        class XlaLinear(nn.Module):
            def __init__(self):
                super(XlaLinear, self).__init__()
                self.linear = nn.Linear(2, 5)

            def forward(self, x):
                return self.linear(x)

        x = torch.rand(4, 2)
        model = XlaLinear()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestConv(TestCase):
    def test(self):

        class XlaConv(nn.Module):
            def __init__(self, bias):
                super(XlaConv, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3, bias=bias)

            def forward(self, x):
                return self.conv(x)

        for bias in [True, False]:
            x = torch.randn(1, 1, 3, 5)
            model = XlaConv(bias)
            out = _xla_run(model, x)
            expected = model(x)
            self.assertEqual(out.data, expected.data)


class TestMaxPool(TestCase):
    def test(self):

        class XlaMaxPool(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 2)

        x = torch.rand(2, 1, 4, 6)
        model = XlaMaxPool()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestLogSoftmax(TestCase):
    def test(self):

        class XlaLogSoftmax(nn.Module):
            def __init__(self, dim):
                super(XlaLogSoftmax, self).__init__()
                self.dim = dim

            def forward(self, x):
                return F.log_softmax(x, self.dim)

        x = torch.rand(4, 2)
        for dim in [0, 1]:
            model = XlaLogSoftmax(dim)
            out = _xla_run(model, x)
            expected = model(x)
            self.assertEqual(out.data, expected.data)


class TestBatchNorm(TestCase):
    def test(self):

        class XlaBatchNorm(nn.Module):
            def __init__(self, training):
                super(XlaBatchNorm, self).__init__()
                if training:
                    self.bn = nn.BatchNorm2d(1)
                else:
                    self.bn = nn.BatchNorm2d(1, track_running_stats=False)

            def forward(self, x):
                return self.bn(x)

        x = torch.rand(1, 1, 5, 7)
        model = XlaBatchNorm(True)
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    run_tests()
