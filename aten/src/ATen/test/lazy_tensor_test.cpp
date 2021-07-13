#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

void LazyFree(void *ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(ptr);
}

void* LazyMalloc(ptrdiff_t size) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  return malloc(size);
}

struct LazyAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    auto* ptr = LazyMalloc(size);
    return {ptr, ptr, &LazyFree, at::DeviceType::XLA};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &LazyFree;
  }
};

void LazyTensorTest(c10::DispatchKey dispatch_key, at::DeviceType device_type) {
  LazyAllocator allocator;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      dispatch_key,
      caffe2::TypeMeta::Make<float>(),
      at::Device(device_type, 0));
  at::Tensor t(std::move(tensor_impl));
  ASSERT_TRUE(t.device() == at::Device(device_type, 0));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(XlaTensorTest, TestNoStorage) {
  LazyTensorTest(DispatchKey::XLA, DeviceType::XLA);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LazyTensorTest, TestNoStorage) {
  LazyTensorTest(DispatchKey::Lazy, DeviceType::Lazy);
}
