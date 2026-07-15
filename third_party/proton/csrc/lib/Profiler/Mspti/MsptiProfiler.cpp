#include "Profiler/Mspti/MsptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/MsptiApi.h"


#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<MsptiProfiler>::ThreadState
    GPUProfiler<MsptiProfiler>::threadState(MsptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<MsptiProfiler>::Correlation::externIdQueue{};

namespace {

std::shared_ptr<Metric>
convertActivityToMetric(const msptiActivityKernel *kernel) {
  if (kernel->start == 0 && kernel->end == 0)
    return nullptr;
  if (kernel->start >= kernel->end)
    return nullptr;
  return std::make_shared<KernelMetric>(
      static_cast<uint64_t>(kernel->start),
      static_cast<uint64_t>(kernel->end),
      /*invocations=*/1,
      static_cast<uint64_t>(kernel->ds.deviceId),
      static_cast<uint64_t>(DeviceType::NPU),
      static_cast<uint64_t>(kernel->ds.streamId));
}

void processActivityKernel(
    MsptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    MsptiProfiler::ApiExternIdSet &apiExternIds,
    std::set<Data *> &dataSet,
    const msptiActivityKernel *kernel,
    uint64_t &outMaxCorrId) {

  auto corrId = kernel->correlationId;
  if (corrId > outMaxCorrId)
    outMaxCorrId = corrId;

  if (!corrIdToExternId.contain(corrId))
    return;

  auto [parentId, numInstances] = corrIdToExternId.at(corrId);

  for (auto *data : dataSet) {
    auto metric = convertActivityToMetric(kernel);
    if (metric)
      data->addMetric(parentId, metric);
  }

  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0)
    corrIdToExternId.erase(corrId);
  else
    corrIdToExternId[corrId].second = numInstances;
}

bool isDriverAPILaunch(msptiCallbackId cbId) {
  return cbId == MSPTI_CBID_RUNTIME_LAUNCH        ||
         cbId == MSPTI_CBID_RUNTIME_AICPU_LAUNCH  ||
         cbId == MSPTI_CBID_RUNTIME_AIV_LAUNCH    ||
         cbId == MSPTI_CBID_RUNTIME_FFTS_LAUNCH   ||
         cbId == MSPTI_CBID_RUNTIME_CPU_LAUNCH;
}

void setRuntimeCallbacks(msptiSubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  mspti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              MSPTI_CB_DOMAIN_RUNTIME, (id))

  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_AICPU_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_AIV_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_FFTS_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_CPU_LAUNCH);
#undef CALLBACK_ENABLE
}

} // namespace

struct MsptiProfiler::MsptiProfilerPimpl
    : public GPUProfiler<MsptiProfiler>::GPUProfilerPimplInterface {

  MsptiProfilerPimpl(MsptiProfiler &profiler)
      : GPUProfiler<MsptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~MsptiProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, msptiCallbackDomain domain,
                         msptiCallbackId cbId, const msptiCallbackData *cbData);

  static constexpr size_t BufferSize = 64 * 1024 * 1024;

  msptiSubscriberHandle subscriber{};
};

void MsptiProfiler::MsptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(std::malloc(BufferSize));
  if (!*buffer)
    throw std::runtime_error("[proton/mspti] allocBuffer: malloc failed");
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void MsptiProfiler::MsptiProfilerPimpl::completeBuffer(uint8_t *buffer,
                                                       size_t /*size*/,
                                                       size_t validSize) {
  fprintf(stderr, "[DEBUG] completeBuffer: called, buffer=%p, validSize=%zu\n", (void*)buffer, validSize);
  MsptiProfiler &profiler = threadState.profiler;
  auto &dataSet = profiler.dataSet;
  if (validSize == 0 || buffer == nullptr) {
    std::free(buffer);
    return;
  }

  uint64_t maxCorrId = 0;
  msptiActivity *record = nullptr;
  msptiResult status;
  do {
    auto status = mspti::activityGetNextRecord<false>(buffer, validSize, &record);
    if (status == MSPTI_SUCCESS) {
      if (record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
        processActivityKernel(
            profiler.correlation.corrIdToExternId,
            profiler.correlation.apiExternIds,
            dataSet,
            reinterpret_cast<const msptiActivityKernel *>(record),
            maxCorrId);
      }
    } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
        throw std::runtime_error("mspti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);
  profiler.correlation.complete(maxCorrId);
}

void MsptiProfiler::MsptiProfilerPimpl::callbackFn(
    void *userData,
    msptiCallbackDomain domain,
    msptiCallbackId cbId,
    const msptiCallbackData *cbData) {
  fprintf(stderr, "[DEBUG] callbackFn: domain=%d, cbId=%d, callbackSite=%d\n",
          (int)domain, (int)cbId, (int)cbData->callbackSite);
  if (domain != MSPTI_CB_DOMAIN_RUNTIME)
    return;
  if (!isDriverAPILaunch(cbId))
    return;

  MsptiProfiler &profiler = threadState.profiler;

  if (cbData->callbackSite == MSPTI_API_ENTER) {
    fprintf(stderr, "[DEBUG] callbackFn: ENTER, correlationId=%lu\n", cbData->correlationId);
    profiler.correlation.correlate(cbData->correlationId, /*numInstances=*/1);
  } else if (cbData->callbackSite == MSPTI_API_EXIT) {
    fprintf(stderr, "[DEBUG] callbackFn: EXIT, correlationId=%lu\n", cbData->correlationId);
    threadState.exitOp();
    profiler.correlation.submit(cbData->correlationId);
  }
}

void MsptiProfiler::MsptiProfilerPimpl::doStart() {
  fprintf(stderr, "[DEBUG] doStart: subscribing mspti callbacks\n");
  auto ret1 = mspti::subscribe<true>(&subscriber, callbackFn, nullptr);
  fprintf(stderr, "[DEBUG] doStart: subscribe ret=%d, subscriber=%p\n", (int)ret1, (void*)subscriber);
  mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  mspti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  setRuntimeCallbacks(subscriber, /*enable=*/true);
  fprintf(stderr, "[DEBUG] doStart: mspti setup complete, subscriber=%p\n", (void*)subscriber);
}

void MsptiProfiler::MsptiProfilerPimpl::doFlush() {
  fprintf(stderr, "[DEBUG] doFlush: flushing activity buffers\n");
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepMs=*/10,
      /*flush=*/[]() { 
        mspti::activityFlushAll<true>(
            /*flag=*/0); 
    });
  mspti::activityFlushAll<true>(/*flag=*/1);
  fprintf(stderr, "[DEBUG] doFlush: flush complete\n");
}

void MsptiProfiler::MsptiProfilerPimpl::doStop() {
  fprintf(stderr, "[DEBUG] doStop: disabling mspti and unsubscribing\n");
  mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  setRuntimeCallbacks(subscriber, /*enable=*/false);
  mspti::unsubscribe<true>(subscriber);
  subscriber = nullptr;
  fprintf(stderr, "[DEBUG] doStop: done\n");
}

MsptiProfiler::MsptiProfiler() {
  pImpl = std::make_unique<MsptiProfilerPimpl>(*this);
}

MsptiProfiler::~MsptiProfiler() = default;

void MsptiProfiler::doSetMode(
  const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (!mode.empty()) {
    throw std::invalid_argument(
        "[PROTON] MsptiProfiler: unsupported mode: " + mode);
  }
}

} // namespace proton