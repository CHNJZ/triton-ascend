#include "Driver/GPU/CannApi.h"
#include "Driver/Dispatch.h"

#include <string>

namespace proton {

namespace cann {

// TODO: 接入 CANN 设备查询接口
// 对标 CudaApi.cpp::cuda::getDevice
// 需要查询的字段：
//   clockRate       → aclrtGetDeviceClockRate 或等价接口
//   memoryClockRate → 待确认 CANN 接口
//   busWidth        → 待确认 CANN 接口
//   numSms          → AI Core 数量，待确认 CANN 接口
//   arch            → 芯片型号字符串，如 "910B2"，待确认 CANN 接口
Device getDevice(uint64_t index) {
  return Device(DeviceType::NPU, index,
                /*clockRate=*/0,
                /*memoryClockRate=*/0,
                /*busWidth=*/0,
                /*numSms=*/0,
                /*arch=*/"ascend");
}

} // namespace cann

} // namespace proton