#ifndef LITE_TENSOR_H
#define LITE_TENSOR_H
#include "mindir_lite_graph.h"
#include "mindir_types.h"

namespace mindspore {
namespace lite {

// ********** Tensor **********
TensorPtr MindIR_Tensor_Create();
TensorPtr MindIR_Tensor_Create(const char *name, DataType data_type, const int32_t *dims, uint32_t dims_size,
                               Format format, const uint8_t *data, uint32_t data_size,
                               const QuantParam *quant_params, uint32_t quant_params_size);
std::string MindIR_Tensor_GetName(ConstTensorPtr tensor);
void MindIR_Tensor_SetName(TensorPtr *tensor, const std::string &name);
DataType MindIR_Tensor_GetDataType(ConstTensorPtr tensor);
void MindIR_Tensor_SetDataType(TensorPtr *tensor, DataType data_type);
std::vector<int32_t> MindIR_Tensor_GetDims(ConstTensorPtr tensor);
void MindIR_Tensor_SetDims(TensorPtr *tensor, const std::vector<int32_t> &dims);
Format MindIR_Tensor_GetFormat(ConstTensorPtr tensor);
void MindIR_Tensor_SetFormat(TensorPtr *tensor, Format format);
void MindIR_Tensor_SetData(TensorPtr *tensor, const std::vector<uint8_t> &data);
std::vector<uint8_t> MindIR_Tensor_GetData(ConstTensorPtr tensor);
std::vector<QuantParam> MindIR_Tensor_GetQuantParams(ConstTensorPtr tensor);
void MindIR_Tensor_SetQuantParams(TensorPtr *tensor, const std::vector<QuantParam> &quant_params);
int32_t MindIR_Tensor_GetNodeType(ConstTensorPtr tensor);

void MindIR_Tensor_Destroy(TensorPtr *tensor);

}  // namespace lite
}  // namespace mindspore

#endif  // LITE_TENSOR_H
