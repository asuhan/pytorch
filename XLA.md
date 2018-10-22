* Install Bazel
* Export environment variable `WITH_XLA=1`. Also, `NO_CUDA=1` for now.
* Run regular install step (`python setup.py install`)
* Run `./third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/rpc/grpc_service_main_cpu --port=51000`
* Optionally: export `TF_XLA_FLAGS='--xla_backend_optimization_level=0'` to improve compilation time.
