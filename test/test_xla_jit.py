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


class TestStack(TestCase):
    def test(self):

        class XlaStack(nn.Module):
            def __init__(self, dim):
                super(XlaStack, self).__init__()
                self.dim = dim

            def forward(self, x, y):
                return torch.stack((x, y), self.dim)

        x = torch.rand(2, 5)
        y = torch.rand(2, 5)
        for dim in [0, 1]:
            model = XlaStack(dim)
            traced_model = torch.jit.trace(x, y)(model)
            out = torch._C._to_xla_module(traced_model)(x, y)
            expected = model(x, y)
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
            def __init__(self, stride, padding):
                super(XlaConv, self).__init__()
                self.conv = nn.Conv2d(10, 100, 5, stride=stride,
                    padding=padding)

            def forward(self, x):
                return self.conv(x)

        for stride in range(1, 4):
            for padding in range(0, 3):
                x = torch.randn(32, 10, 28, 28)
                model = XlaConv(stride, padding)
                out = _xla_run(model, x)
                expected = model(x)
                self.assertEqual(out.data, expected.data)


class TestMaxPool(TestCase):
    def test(self):

        class XlaMaxPool(nn.Module):
            def __init__(self, stride, padding):
                super(XlaMaxPool, self).__init__()
                self.stride = stride
                self.padding = padding

            def forward(self, x):
                return F.max_pool2d(x, 3, stride=self.stride,
                    padding=self.padding)

        x = torch.rand(1, 64, 112, 112)
        for stride in [None, 2]:
            for padding in [0, 1]:
                model = XlaMaxPool(stride, padding)
                out = _xla_run(model, x)
                expected = model(x)
                self.assertEqual(out.data, expected.data)


class TestAvgPool(TestCase):
    def test(self):

        class XlaAvgPool(nn.Module):
            def __init__(self, stride, padding, count_include_pad):
                super(XlaAvgPool, self).__init__()
                self.stride = stride
                self.padding = padding
                self.count_include_pad = count_include_pad

            def forward(self, x):
                return F.avg_pool2d(x, 2, self.stride, self.padding, False, self.count_include_pad)

        x = torch.rand(1, 1, 3, 3)
        for stride in [1, 2]:
            for padding in [0, 1]:
                for count_include_pad in [False, True]:
                    model = XlaAvgPool(stride, padding, count_include_pad)
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

        x = torch.rand(5, 3, 4, 2)
        for dim in range(0, x.dim()):
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
                    self.bn = nn.BatchNorm2d(3)
                else:
                    self.bn = nn.BatchNorm2d(3, track_running_stats=False)

            def forward(self, x):
                return self.bn(x)

        x = torch.rand(14, 3, 5, 7)
        model = XlaBatchNorm(True)
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestMNIST(TestCase):
    def test(self):

        class XlaMNIST(nn.Module):
            def __init__(self):
                super(XlaMNIST, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2(x), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        batch_size = 32
        x = torch.randn(batch_size, 1, 28, 28)
        model = XlaMNIST()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestAvgPoolGrad(TestCase):

    def checkGrad(self, model, x, grad_outputs):
        traced_model = torch.jit.trace(x)(model)
        fwd = traced_model._get_method('forward')
        torch._C._jit_pass_decompose_addmm(fwd.graph)

        # for now, all inputs require grad, not just parameters
        inputs_require_grad = [True for i in fwd.graph.inputs()]
        gradient = torch._C._jit_differentiate(fwd.graph, inputs_require_grad)

        defined_df_inputs = [True for i in gradient.df.inputs()] # all df inputs are defined (usual case)
        torch._C._jit_pass_specialize_undef(gradient.df, defined_df_inputs)

        exec_f = torch._C.GraphExecutor(gradient.f, True)
        exec_df = torch._C.GraphExecutor(gradient.df, True)

        # forward function
        inputs = [x]
        raw_outputs = exec_f(*inputs)

        if isinstance(raw_outputs, torch.Tensor):
            raw_outputs = [raw_outputs]
        outputs = raw_outputs[:gradient.f_real_outputs]

        raw_grad_outputs = []
        raw_grad_outputs += grad_outputs
        raw_grad_outputs += [inputs[i] for i in gradient.df_input_captured_inputs]
        raw_grad_outputs += [raw_outputs[i] for i in gradient.df_input_captured_outputs]

        grad_input = exec_df(*raw_grad_outputs)
        # backward with XLA
        traced_backward = torch._C._to_xla_module_grad(traced_model, gradient.df)
        xla_grad_input = traced_backward(*raw_grad_outputs)

        # forward + backward with regular autograd / torch
        out_groundtruth = model(x)
        out_groundtruth.backward(*grad_outputs)
        self.assertEqual(outputs[0], out_groundtruth)
        self.assertEqual(grad_input, inputs[0].grad)
        self.assertEqual(grad_input, xla_grad_input)

    def test_avgpool(self):
        class AvgPoolGrad(nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, 2, stride=1)

        model = AvgPoolGrad()
        x = torch.randn(4, 1, 28, 28, requires_grad=True)
        grad_outputs = [torch.randn(4, 1, 27, 27)] # random grad_output
        self.checkGrad(model, x, grad_outputs)

    def test_threshold(self):
        class ThresholdPoolGrad(nn.Module):
            def __init__(self):
                super(ThresholdPoolGrad, self).__init__()
                self.threshold = nn.Threshold(0.4, 20)

            def forward(self, x):
                return self.threshold(x)

        model = ThresholdPoolGrad()
        x = torch.randn(4, 2, requires_grad=True)
        grad_outputs = [torch.randn(4, 2)] # random grad_output
        self.checkGrad(model, x, grad_outputs)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    run_tests()
