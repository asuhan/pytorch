import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common import TestCase, run_tests
import unittest


def _xla_run(model, input):
    traced_model = torch.jit.trace(input)(model)
    input_xla = torch._C.XLATensor(input)
    xla_model = torch._C.XlaModule(traced_model, [input], backward=False)
    output_xla = xla_model(input_xla)
    return output_xla.to_tensor()


def _forward_passes(graph):
    torch._C._jit_pass_decompose_addmm(graph)
    torch._C._jit_pass_unwrap_buffered_functions(graph)
    torch._C._jit_pass_constant_fold(graph)
    torch._C._jit_pass_dce(graph)


def _backward_passes(graph):
    defined = [True for i in graph.inputs()]
    torch._C._jit_pass_specialize_undef(graph, defined)
    torch._C._jit_pass_constant_fold(graph)
    torch._C._jit_pass_dce(graph)

class TestMulAdd(TestCase):
    def test(self):

        class XlaMulAdd(nn.Module):
            def forward(self, x, y):
                return x * y + y

        x = torch.rand(3, 5)
        y = torch.rand(3, 5)
        model = XlaMulAdd()
        traced_model = torch.jit.trace(x, y)(model)
        xla_model = torch._C.XlaModule(traced_model, [x, y], False)
        output_xla = xla_model(torch._C.XLATensor(x), torch._C.XLATensor(y))
        expected = model(x, y)
        self.assertEqual(output_xla.to_tensor().data, expected.data)


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
            xla_model = torch._C.XlaModule(traced_model, [x, y], False)
            output_xla = xla_model(torch._C.XLATensor(x), torch._C.XLATensor(y))
            expected = model(x, y)
            self.assertEqual(output_xla.to_tensor().data, expected.data)


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
            def __init__(self, stride, padding, bias):
                super(XlaConv, self).__init__()
                self.conv = nn.Conv2d(10, 100, 5, stride=stride,
                    padding=padding, bias=bias)

            def forward(self, x):
                return self.conv(x)

        for stride in range(1, 4):
            for padding in range(0, 3):
                for bias in [True, False]:
                    x = torch.randn(32, 10, 28, 28)
                    model = XlaConv(stride, padding, bias)
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
        for stride in [1, 2, None]:
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

class TestMNIST(TestCase):
    def test(self):
        batch_size = 32
        x = torch.randn(batch_size, 1, 28, 28)
        model = XlaMNIST()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqual(out.data, expected.data)


class TestSum(TestCase):
    def test(self):

        class XlaSum(nn.Module):
            def __init__(self, dim):
                super(XlaSum, self).__init__()
                self.dim = dim

            def forward(self, x):
                return x.sum(dim=self.dim)

        x = torch.randn(2, 3, 4, 6)
        for dim in range(0, x.dim()):
            model = XlaSum(dim)
            out = _xla_run(model, x)
            expected = model(x)
            self.assertEqual(out.data, expected.data)


def _maybe_list(t):
    if isinstance(t, torch.Tensor):
        t = [t]
    return t

def _random_like(tensor_list):
    random_tensors = []
    for o in tensor_list:
        if o.dtype == torch.float32 or o.dtype == torch.float64:
            random_tensors += [torch.randn(*o.shape, dtype=o.dtype)]
        elif o.dtype == torch.int64:
            # TODO remove this, we shouldn't be needing to pass random_tensor for long types
            random_tensors += [torch.empty_like(o)]
        else:
            raise RuntimeError("Unsupported type: ", o.dtype)
    return random_tensors

def _zeros_like(tensor_list):
    zeros_tensors = []
    for o in tensor_list:
        if o.dtype == torch.float32 or o.dtype == torch.float64:
            zeros_tensors += [torch.zeros(*o.shape, dtype=o.dtype)]
        elif o.dtype == torch.int64:
            # TODO remove this, we shouldn't be needing to pass zeros_tensor for long types
            zeros_tensors += [torch.zeros_like(o)]
        else:
            raise RuntimeError("Unsupported type: ", o.dtype)
    return zeros_tensors

class TestGradients(TestCase):
    def checkGrad(self, model, inputs, grad_outputs='random', xla=True):

        # Trace and symbolically differentiate
        traced_model = torch.jit.trace(*inputs)(model)
        fwd = traced_model._get_method('forward')
        _forward_passes(fwd.graph)

        inputs_params = inputs + list(model.parameters())
        inputs_params_buffers = inputs + list(fwd.params())

        inputs_require_grad = [i.requires_grad for i in inputs_params_buffers]
        gradient = torch._C._jit_differentiate(fwd.graph, inputs_require_grad)
        _forward_passes(gradient.f)
        _backward_passes(gradient.df)

        ##############################################################
        # Run forward and backwarg graphs via jit interpreter
        exec_f = torch._C.GraphExecutor(gradient.f, False)
        exec_df = torch._C.GraphExecutor(gradient.df, False)

        # forward function
        raw_outputs = exec_f(*inputs_params_buffers)
        raw_outputs = _maybe_list(raw_outputs)
        outputs = raw_outputs[:gradient.f_real_outputs]

        if grad_outputs == 'random':
            grad_outputs = _random_like(outputs) + _zeros_like(raw_outputs[gradient.f_real_outputs:])

        raw_grad_outputs = []
        raw_grad_outputs += grad_outputs
        raw_grad_outputs += [inputs_params_buffers[i] for i in gradient.df_input_captured_inputs]
        raw_grad_outputs += [raw_outputs[i] for i in gradient.df_input_captured_outputs]

        grad_inputs = exec_df(*raw_grad_outputs)
        grad_inputs = _maybe_list(grad_inputs)

        ##############################################################
        # backward with XLA
        if xla:
            xla_model = torch._C.XlaModule(traced_model, inputs)
            inputs_xla = [torch._C.XLATensor(input) for input in inputs]
            xla_model(*inputs_xla)
            grads_output_xla = [torch._C.XLATensor(grad_output) for grad_output in grad_outputs[:gradient.f_real_outputs]]
            xla_model.backward(*grads_output_xla)
            grad_inputs_xla = [input_xla.grad.to_tensor() for input_xla in inputs_xla]
            grad_inputs_xla.extend([p.grad.to_tensor() for p in xla_model.parameters()])
        ##############################################################
        # forward + backward with regular autograd / torch
        outputs_gt = model(*inputs)
        outputs_gt = _maybe_list(outputs_gt)
        grad_inputs_gt = torch.autograd.grad(outputs_gt,
                                             inputs_params,
                                             grad_outputs,
                                             only_inputs=True)
        for out_jit, out_autograd in zip(outputs, outputs_gt):
            self.assertEqual(out_jit, out_autograd)

        for grad_input_jit, grad_input_autograd in zip(grad_inputs, grad_inputs_gt):
            self.assertEqual(grad_input_jit, grad_input_autograd)

        # TODO: test buffers as well (running_mean, etc.)

        if xla:
            for i, (grad_input_jit, grad_input_xla) in enumerate(zip(grad_inputs, grad_inputs_xla)):
                self.assertEqual(grad_input_jit, grad_input_xla, 1e-3)

    def test_avgpool(self):
        class AvgPoolGrad(nn.Module):
            def __init__(self, stride, padding, count_include_pad):
                super(AvgPoolGrad, self).__init__()
                self.stride = stride
                self.padding = padding
                self.count_include_pad = count_include_pad

            def forward(self, x):
                return F.avg_pool2d(x, 2, self.stride, self.padding, False, self.count_include_pad)

        for stride in [1, 2]:
            for padding in [0, 1]:
                for count_include_pad in [False, True]:
                    model = AvgPoolGrad(stride, padding, count_include_pad)
                    inputs = [torch.randn(4, 1, 28, 28, requires_grad=True)]
                    self.checkGrad(model, inputs, xla=True)

    def test_threshold(self):
        class ThresholdPoolGrad(nn.Module):
            def __init__(self):
                super(ThresholdPoolGrad, self).__init__()
                self.threshold = nn.Threshold(0.4, 20)

            def forward(self, x):
                return self.threshold(x)

        model = ThresholdPoolGrad()
        inputs = [torch.randn(4, 2, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)


    def test_maxpool(self):
        class MaxPoolGrad(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 2)

        model = MaxPoolGrad()
        inputs = [torch.randn(4, 1, 28, 28, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    def test_conv2d(self):
        for ichans in [1, 15, 32]:
            for ochans in [1, 13, 32]:
                for size in [1, 2, 3]:
                    for stride in [1, 2]:
                        for padding in [0, 1]:
                            for bias in [True, False]:
                                # TODO: dilation, groups, transpose
                                model = nn.Conv2d(ichans, ochans, size, stride, padding, bias=bias)
                                inputs = [torch.randn(4, ichans, 28, 28, requires_grad=True)]
                                self.checkGrad(model, inputs, xla=True)

    def test_batchnorm2d(self):
        for chans in [1, 15, 32]:
            for eps in [1e-5, 1e-3, 1e-2]:
                # TODO: momentum, training, affine
                model = nn.BatchNorm2d(chans, eps=eps)
                inputs = [torch.randn(4, chans, 28, 28, requires_grad=True)]
                self.checkGrad(model, inputs, xla=True)

    def test_logsoftmax(self):
        for dim in [0, 1]: # todo test 3d as well
            for batch in [1, 3, 4]:
                class LSMGrad(nn.Module):
                    def forward(self, x):
                        return F.log_softmax(x, dim)

                model = LSMGrad()
                inputs = [torch.randn(batch, 9, requires_grad=True)]
                self.checkGrad(model, inputs, xla=True)

    def test_mnist(self):
        model = XlaMNIST()
        inputs = [torch.randn(4, 1, 28, 28, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    def test_resnet(self):
        import torchvision
        model = torchvision.models.resnet50()
        inputs = [torch.randn(4, 3, 224, 224, requires_grad=True)]
        self.checkGrad(model, inputs, xla=False)


class TestOptimizer(TestCase):
    def test_add_mul(self):
        orig_x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        orig_y = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        x = orig_x
        y = orig_y
        xla_x = torch._C.XLATensor(orig_x)
        xla_y = torch._C.XLATensor(orig_y)
        self.assertEqual(x.add_(2, y).mul_(y), xla_x.add_(2, xla_y).mul_(xla_y).to_tensor())
        self.assertEqual(x.add_(y).mul_(y), xla_x.add_(xla_y).mul_(xla_y).to_tensor())

    def checkSgd(self, lr, momentum, weight_decay, nsteps, do_zero_grad):
        input = torch.randn(4, 4, requires_grad=True)
        input_xla = torch._C.XLATensor(input)
        model = nn.Linear(4, 20)
        traced_model = torch.jit.trace(input)(model)
        xla_model = torch._C.XlaModule(traced_model, [input])
        xla_optimizer = optim.XlaSGD(xla_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        xla_model(input_xla)
        output = model(input)
        grad_output = torch.randn(*output.shape) # random gradients
        grad_output_xla = torch._C.XLATensor(grad_output)
        output.backward(grad_output)
        xla_model.backward(grad_output_xla)
        if do_zero_grad:
            optimizer.zero_grad()
            xla_optimizer.zero_grad()
        for _ in range(0, nsteps):
            xla_optimizer.step()
            optimizer.step()
            xla_updated_params = [p.to_tensor().data for p in xla_model.parameters()]
            updated_params = [p.data for p in model.parameters()]
            self.assertEqual(xla_updated_params, updated_params);

    def test_sgd(self):
        for weight_decay in [0, 5e-4]:
            self.checkSgd(lr=0.1, momentum=0, weight_decay=weight_decay, nsteps=1, do_zero_grad=True)
            self.checkSgd(lr=0.1, momentum=0, weight_decay=weight_decay, nsteps=2, do_zero_grad=False)
            self.checkSgd(lr=0.1, momentum=0.5, weight_decay=weight_decay, nsteps=1, do_zero_grad=True)
            self.checkSgd(lr=0.1, momentum=0.5, weight_decay=weight_decay, nsteps=2, do_zero_grad=False)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    run_tests()
