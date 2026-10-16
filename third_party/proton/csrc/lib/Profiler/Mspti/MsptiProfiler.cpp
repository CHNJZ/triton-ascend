#include "Profiler/Mspti/MsptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/MsptiApi.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<MsptiProfiler>::ThreadState
    GPUProfiler<MsptiProfiler>::threadState(MsptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<MsptiProfiler>::Correlation::externIdQueue{};

namespace {

thread_local std::vector<bool> callbackOwnsOp;
thread_local bool callbackCreatingOp = false;
// Triton launch hooks and MSPTI runtime callbacks can run on different
// threads. Preserve launch order so callbacks can reuse hook-created ops.
std::deque<uint64_t> pendingExternalIds;
std::mutex pendingExternalIdsMutex;
std::unordered_map<uint64_t, std::string> callbackKernelNames;
std::mutex callbackKernelNamesMutex;
std::atomic<size_t> pendingKernelActivities{0};

bool msptiDebugEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("PROTON_MSPTI_DEBUG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

void msptiDebugLog(const char *format, ...) {
  if (!msptiDebugEnabled())
    return;
  const auto threadId =
      std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::fprintf(stderr, "[PROTON_MSPTI_DEBUG][tid=%zu] ", threadId);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fputc('\n', stderr);
  std::fflush(stderr);
}

class ScopedCallbackState {
public:
  ScopedCallbackState() : previousState(ContextSource::getThreadState()) {
    msptiDebugLog("state-bridge begin previous=%d",
                  previousState.has_value());
    if (previousState.has_value())
      return;
    auto activeState = ContextSource::getActiveState();
    msptiDebugLog("state-bridge active=%d", activeState.has_value());
    if (!activeState.has_value())
      return;
    ContextSource::setThreadState(std::move(activeState));
    installedState = true;
    msptiDebugLog("state-bridge installed");
  }

  ~ScopedCallbackState() {
    if (installedState) {
      msptiDebugLog("state-bridge restore");
      ContextSource::setThreadState(std::move(previousState));
    }
  }

private:
  std::optional<Context> previousState;
  bool installedState{false};
};

bool isPrintableAscii(const std::string &value) {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character >= 0x20 && character <= 0x7e;
         });
}

std::shared_ptr<Metric>
convertActivityToMetric(const msptiActivityKernel *kernel) {
  if (kernel->start == 0 && kernel->end == 0)
    return nullptr;
  if (kernel->start >= kernel->end)
    return nullptr;
  return std::make_shared<KernelMetric>(
      static_cast<uint64_t>(kernel->start), static_cast<uint64_t>(kernel->end),
      /*invocations=*/1, static_cast<uint64_t>(kernel->ds.deviceId),
      static_cast<uint64_t>(DeviceType::NPU),
      static_cast<uint64_t>(kernel->ds.streamId));
}

void processActivityKernel(MsptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                           MsptiProfiler::ApiExternIdSet &apiExternIds,
                           std::set<Data *> &dataSet,
                           const msptiActivityKernel *kernel,
                           const std::string &activityKernelName,
                           uint64_t &outMaxCorrId) {

  auto corrId = kernel->correlationId;
  msptiDebugLog(
      "process-kernel begin corr=%llu start=%llu end=%llu name-size=%zu",
      static_cast<unsigned long long>(corrId),
      static_cast<unsigned long long>(kernel->start),
      static_cast<unsigned long long>(kernel->end), activityKernelName.size());
  if (corrId > outMaxCorrId)
    outMaxCorrId = corrId;

  if (!corrIdToExternId.contain(corrId)) {
    msptiDebugLog("process-kernel missing-correlation corr=%llu",
                  static_cast<unsigned long long>(corrId));
    return;
  }

  auto [parentId, numInstances] = corrIdToExternId.at(corrId);
  const auto isApiOp = apiExternIds.contain(parentId);
  msptiDebugLog("process-kernel mapped corr=%llu external=%zu api-op=%d",
                static_cast<unsigned long long>(corrId), parentId, isApiOp);
  std::string apiKernelName{"kernel"};
  if (isApiOp) {
    std::lock_guard<std::mutex> lock(callbackKernelNamesMutex);
    auto nameIt = callbackKernelNames.find(parentId);
    if (nameIt != callbackKernelNames.end()) {
      apiKernelName = std::move(nameIt->second);
      callbackKernelNames.erase(nameIt);
    }
    if (isPrintableAscii(activityKernelName))
      apiKernelName = activityKernelName;
  }

  for (auto *data : dataSet) {
    auto metric = convertActivityToMetric(kernel);
    if (!metric)
      continue;
    auto scopeId = parentId;
    if (isApiOp)
      scopeId = data->addOp(parentId, apiKernelName);
    data->addMetric(scopeId, metric);
    msptiDebugLog("process-kernel metric-added external=%zu scope=%zu",
                  parentId, scopeId);
  }

  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0)
    corrIdToExternId.erase(corrId);
  else
    corrIdToExternId[corrId].second = numInstances;

  auto pending = pendingKernelActivities.load(std::memory_order_relaxed);
  while (pending != 0 && !pendingKernelActivities.compare_exchange_weak(
                             pending, pending - 1, std::memory_order_relaxed)) {
  }
  msptiDebugLog("process-kernel end corr=%llu pending=%zu",
                static_cast<unsigned long long>(corrId),
                pendingKernelActivities.load(std::memory_order_relaxed));
}

bool isDriverAPILaunch(msptiCallbackId cbId) {
  return cbId == MSPTI_CBID_RUNTIME_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_AICPU_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_AIV_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_FFTS_LAUNCH ||
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
  msptiDebugLog("buffer-alloc ptr=%p size=%zu", static_cast<void *>(*buffer),
                *bufferSize);
}

void MsptiProfiler::MsptiProfilerPimpl::completeBuffer(uint8_t *buffer,
                                                       size_t /*size*/,
                                                       size_t validSize) {
  msptiDebugLog("buffer-complete begin ptr=%p valid=%zu",
                static_cast<void *>(buffer), validSize);
  MsptiProfiler &profiler = threadState.profiler;
  auto &dataSet = profiler.dataSet;
  if (validSize == 0 || buffer == nullptr) {
    std::free(buffer);
    return;
  }

  uint64_t maxCorrId = 0;
  msptiActivity *record = nullptr;
  std::vector<msptiActivityKernel> kernelRecords;
  std::vector<std::string> kernelNames;
  do {
    auto status =
        mspti::activityGetNextRecord<false>(buffer, validSize, &record);
    if (status == MSPTI_SUCCESS) {
      msptiDebugLog("buffer-record kind=%u ptr=%p",
                    static_cast<unsigned int>(record->kind),
                    static_cast<void *>(record));
      if (record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
        auto *kernel = reinterpret_cast<const msptiActivityKernel *>(record);
        msptiDebugLog(
            "buffer-kernel corr=%llu start=%llu end=%llu name-ptr=%p",
            static_cast<unsigned long long>(kernel->correlationId),
            static_cast<unsigned long long>(kernel->start),
            static_cast<unsigned long long>(kernel->end),
            const_cast<char *>(kernel->name));
        kernelRecords.push_back(*kernel);
        // The name pointer is only stable while MSPTI owns the current record.
        msptiDebugLog("buffer-kernel name-copy begin");
        kernelNames.emplace_back(kernel->name == nullptr ? "" : kernel->name);
        msptiDebugLog("buffer-kernel name-copy end size=%zu",
                      kernelNames.back().size());
      } else if (record->kind == MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        auto *external =
            reinterpret_cast<const msptiActivityExternalCorrelation *>(record);
        if (external->externalKind == MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
          msptiDebugLog("buffer-external corr=%llu external=%llu",
                        static_cast<unsigned long long>(
                            external->correlationId),
                        static_cast<unsigned long long>(external->externalId));
          profiler.correlation.corrIdToExternId[external->correlationId] = {
              static_cast<size_t>(external->externalId), 1};
        }
      }
    } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("mspti::activityGetNextRecord failed");
    }
  } while (true);

  // MSPTI does not guarantee that external-correlation records precede kernel
  // records in a completed buffer. Build all mappings before consuming kernels.
  for (size_t index = 0; index < kernelRecords.size(); ++index) {
    processActivityKernel(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet,
                          &kernelRecords[index], kernelNames[index], maxCorrId);
  }

  std::free(buffer);
  profiler.correlation.complete(maxCorrId);
  msptiDebugLog("buffer-complete end max-corr=%llu",
                static_cast<unsigned long long>(maxCorrId));
}

void MsptiProfiler::MsptiProfilerPimpl::callbackFn(
    void * /*userData*/, msptiCallbackDomain domain, msptiCallbackId cbId,
    const msptiCallbackData *cbData) {
  if (domain != MSPTI_CB_DOMAIN_RUNTIME)
    return;
  if (!isDriverAPILaunch(cbId))
    return;
  if (cbData == nullptr) {
    msptiDebugLog("callback null-data cbid=%u", cbId);
    return;
  }

  msptiDebugLog(
      "callback begin cbid=%u site=%u corr=%llu data=%p symbol=%p function=%p "
      "corr-data=%p owns-depth=%zu",
      cbId, static_cast<unsigned int>(cbData->callbackSite),
      static_cast<unsigned long long>(cbData->correlationId),
      const_cast<msptiCallbackData *>(cbData),
      const_cast<char *>(cbData->symbolName),
      const_cast<char *>(cbData->functionName),
      static_cast<void *>(cbData->correlationData), callbackOwnsOp.size());

  if (cbData->callbackSite == MSPTI_API_ENTER) {
    uint64_t externalId = Scope::DummyScopeId;
    bool hasHookOp = false;
    {
      std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
      if (!pendingExternalIds.empty()) {
        externalId = pendingExternalIds.front();
        pendingExternalIds.pop_front();
        hasHookOp = true;
      }
    }
    msptiDebugLog("callback enter hook-op=%d external=%llu", hasHookOp,
                  static_cast<unsigned long long>(externalId));

    bool ownsOp = false;
    if (!hasHookOp) {
      msptiDebugLog("callback no-hook state-bridge begin");
      ScopedCallbackState callbackState;
      callbackCreatingOp = true;
      msptiDebugLog("callback no-hook enter-op begin");
      threadState.enterOp();
      msptiDebugLog("callback no-hook enter-op end op=%zu", threadState.opId);
      callbackCreatingOp = false;
      externalId = threadState.opId;
      // Some CANN 9.2 runtimes return a non-dereferenceable symbolName for
      // MSPTI_CBID_RUNTIME_LAUNCH. The kernel activity name is preferred later.
      std::string callbackKernelName{"kernel"};
      msptiDebugLog("callback no-hook installed fallback kernel name");
      std::lock_guard<std::mutex> lock(callbackKernelNamesMutex);
      callbackKernelNames[externalId] = std::move(callbackKernelName);
      ownsOp = true;
      msptiDebugLog("callback no-hook op-ready external=%llu",
                    static_cast<unsigned long long>(externalId));
    }
    msptiDebugLog("callback push-correlation begin external=%llu",
                  static_cast<unsigned long long>(externalId));
    mspti::activityPushExternalCorrelationId<true>(
        MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, externalId);
    msptiDebugLog("callback push-correlation end external=%llu",
                  static_cast<unsigned long long>(externalId));
    callbackOwnsOp.push_back(ownsOp);
    pendingKernelActivities.fetch_add(1, std::memory_order_relaxed);
    msptiDebugLog("callback enter end owns=%d depth=%zu pending=%zu", ownsOp,
                  callbackOwnsOp.size(),
                  pendingKernelActivities.load(std::memory_order_relaxed));
  } else if (cbData->callbackSite == MSPTI_API_EXIT) {
    msptiDebugLog("callback exit begin depth=%zu", callbackOwnsOp.size());
    if (!callbackOwnsOp.empty()) {
      uint64_t externalId = 0;
      msptiDebugLog("callback pop-correlation begin");
      mspti::activityPopExternalCorrelationId<true>(
          MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &externalId);
      msptiDebugLog("callback pop-correlation end external=%llu owns=%d",
                    static_cast<unsigned long long>(externalId),
                    callbackOwnsOp.back());
      if (callbackOwnsOp.back()) {
        msptiDebugLog("callback exit-op begin op=%zu", threadState.opId);
        threadState.exitOp();
        msptiDebugLog("callback exit-op end op=%zu", threadState.opId);
      }
      callbackOwnsOp.pop_back();
    }
    msptiDebugLog("callback exit end depth=%zu", callbackOwnsOp.size());
  }
}

void MsptiProfiler::MsptiProfilerPimpl::doStart() {
  msptiDebugLog("start begin");
  {
    std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
    pendingExternalIds.clear();
  }
  profiler.correlation.corrIdToExternId.clear();
  profiler.correlation.apiExternIds.clear();
  {
    std::lock_guard<std::mutex> lock(callbackKernelNamesMutex);
    callbackKernelNames.clear();
  }
  pendingKernelActivities.store(0, std::memory_order_relaxed);
  mspti::subscribe<true>(&subscriber, callbackFn, nullptr);
  msptiDebugLog("start subscribed handle=%p", static_cast<void *>(subscriber));
  mspti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  msptiDebugLog("start buffer-callbacks-registered");
  mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  msptiDebugLog("start kernel-enabled");
  mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  msptiDebugLog("start external-correlation-enabled");
  setRuntimeCallbacks(subscriber, /*enable=*/true);
  msptiDebugLog("start runtime-callbacks-enabled");
}

void MsptiProfiler::MsptiProfilerPimpl::doFlush() {
  msptiDebugLog("flush begin pending=%zu",
                pendingKernelActivities.load(std::memory_order_relaxed));
  mspti::activityFlushAll<true>(/*flag=*/0);
  msptiDebugLog("flush normal-complete pending=%zu",
                pendingKernelActivities.load(std::memory_order_relaxed));
  if (pendingKernelActivities.load(std::memory_order_relaxed) != 0) {
    // MSPTI 26.1 flushes its host ActivityBuffer but does not flush the
    // device profiling channel. Cycling only KERNEL forces those SOC records
    // through KernelParser while the Proton Data objects are still registered.
    msptiDebugLog("flush kernel-cycle disable");
    mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
    msptiDebugLog("flush kernel-cycle intermediate-flush");
    mspti::activityFlushAll<true>(/*flag=*/0);
    msptiDebugLog("flush kernel-cycle enable");
    mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  }
  msptiDebugLog("flush forced begin");
  mspti::activityFlushAll<true>(/*flag=*/1);
  msptiDebugLog("flush end pending=%zu",
                pendingKernelActivities.load(std::memory_order_relaxed));
}

void MsptiProfiler::MsptiProfilerPimpl::doStop() {
  msptiDebugLog("stop begin");
  mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  msptiDebugLog("stop external-correlation-disabled");
  mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  msptiDebugLog("stop kernel-disabled");
  setRuntimeCallbacks(subscriber, /*enable=*/false);
  msptiDebugLog("stop runtime-callbacks-disabled");
  mspti::unsubscribe<true>(subscriber);
  msptiDebugLog("stop unsubscribed");
  subscriber = nullptr;
  profiler.correlation.corrIdToExternId.clear();
  profiler.correlation.apiExternIds.clear();
  {
    std::lock_guard<std::mutex> lock(callbackKernelNamesMutex);
    callbackKernelNames.clear();
  }
  pendingKernelActivities.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
  pendingExternalIds.clear();
  msptiDebugLog("stop end");
}

MsptiProfiler::MsptiProfiler() {
  pImpl = std::make_unique<MsptiProfilerPimpl>(*this);
}

MsptiProfiler::~MsptiProfiler() = default;

void MsptiProfiler::startOp(const Scope &scope) {
  msptiDebugLog("start-op begin scope=%zu name-size=%zu callback-created=%d",
                scope.scopeId, scope.name.size(), callbackCreatingOp);
  GPUProfiler<MsptiProfiler>::startOp(scope);
  if (callbackCreatingOp) {
    msptiDebugLog("start-op callback-created scope=%zu", scope.scopeId);
    return;
  }
  std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
  pendingExternalIds.push_back(static_cast<uint64_t>(scope.scopeId));
  msptiDebugLog("start-op queued scope=%zu queue-size=%zu", scope.scopeId,
                pendingExternalIds.size());
}

void MsptiProfiler::stopOp(const Scope &scope) {
  msptiDebugLog("stop-op scope=%zu", scope.scopeId);
  GPUProfiler<MsptiProfiler>::stopOp(scope);
}

void MsptiProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] MsptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
