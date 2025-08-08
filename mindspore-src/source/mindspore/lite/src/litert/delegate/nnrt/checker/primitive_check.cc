#include <string>
#include <vector>
#include "primitive_check.h"
#include "dtype/type_id.h"
#include "src/litert/weight_decoder.h"
#include "src/common/log.h"
#include "src/common/utils.h"
namespace mindspore {
namespace lite {
namespace {
bool NeedBitUppackCheck(const schema::Tensor &src_tensor) {
  if (src_tensor.enableHuffmanCode()) {
    return true;
  }
  bool need_bit_unpack = src_tensor.quantParams() != nullptr && src_tensor.quantParams()->size() > 0 &&
                         src_tensor.quantParams()->Get(0) != nullptr;
  if (need_bit_unpack) {
    auto num_bits = src_tensor.quantParams()->Get(0)->numBits();
    need_bit_unpack = ((num_bits >= kBitNum1 && num_bits < kBitNum8) || (num_bits > kBitNum8 && num_bits < kBitNum16));
  }

  return need_bit_unpack;
}
int DecompressTensor(const schema::Tensor &src_tensor) {
  if (src_tensor.weightQuantCompressType() == schema::WeightQuantCompressType_FSE ||
      src_tensor.weightQuantCompressType() == schema::WeightQuantCompressType_INDEXING ||
      src_tensor.weightQuantCompressType() == schema::WeightQuantCompressType_SPARSE) {
    return RET_NOT_SUPPORT;
  }
  if (!NeedBitUppackCheck(src_tensor)) {
    return RET_NO_CHANGE;
  }
  MS_LOG(ERROR) << "DecompressTensor Error.";
  return RET_ERROR;
}
}  // namespace

Status CheckTensorSupported(const schema::Tensor *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr, which type is Tensor.";
    return mindspore::kLiteSuccessExit;
  }

  int32_t data_type = primitive->dataType();
  if (data_type <= kTypeUnknown || data_type >= kMonadTypeEnd) {
    MS_LOG(ERROR) << "invalid data type. " << data_type;
    return mindspore::kLiteSuccessExit;
  }

  if (primitive->dims() == nullptr) {
    MS_LOG(DEBUG) << "Dims of tensor is nullptr";
  }

  if (data_type == kObjectTypeTensorType) {
    MS_LOG(ERROR) << "Not support TensorList.";
    return mindspore::kLiteNotSupport;
  }

  if (primitive->data() == nullptr || primitive->data()->size() <= 0) {
    MS_LOG(DEBUG) << "No valid data converted.";
    return mindspore::kSuccess;
  } else {
    auto ret = DecompressTensor(*primitive);
    if (ret == RET_NO_CHANGE) {
    } else {
      MS_LOG(ERROR) << "Not support Decompress Tensor.";
      return mindspore::kLiteNotSupport;
    }
  }
  return mindspore::kSuccess;
  ;
}
}  // namespace lite
}  // namespace mindspore
