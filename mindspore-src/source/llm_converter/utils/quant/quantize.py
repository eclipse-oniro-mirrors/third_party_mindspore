'''model quantize'''
from utils.quant.quant_config import ModelConfig
from utils.duplicate_shared_initializer import duplicate_shared_initializers
import onnx
from onnxslim import slim
from onnx import helper, shape_inference, TensorProto
import numpy as np
import ctypes
import os
import argparse
try:
    dll = ctypes.cdll.LoadLibrary
    lib_q4 = dll(os.path.join(os.path.dirname(__file__), 'q4_weight_quant.so'))
except:
    raise ValueError(
        "Load q4_weight_quant.so failed. Compile with `gcc -o q4_weight_quant.so -shared -fPIC q4_weight_quant.cc`"
    )

try:
    dll = ctypes.cdll.LoadLibrary
    lib_q2 = dll(os.path.join(os.path.dirname(__file__), 'q2_weight_quant.so'))
except:
    raise ValueError(
        "Load q2_weight_quant.so failed. Compile with `gcc -o q2_weight_quant.so -shared -fPIC q2_weight_quant.cc`"
    )

try:
    dll = ctypes.cdll.LoadLibrary
    lib_q4_gp32 = dll(os.path.join(os.path.dirname(
        __file__), 'q4_weight_quant_gp32.so'))
except:
    raise ValueError(
        "Load q4_weight_quant_gp32.so failed. Compile with `g++ -o q4_weight_quant_gp32.so -shared -fPIC q4_weight_quant_gp32.cc`"
    )


def custom_op_infer_shape(graph, chunk_size, max_seq_len, num_kv_heads, num_q_heads, dim, is_prefill):
    '''infer shape for custom operators'''
    graph_output = {vi.name: vi for vi in graph.output}

    for node in graph.node:
        if node.op_type == 'MsScatterND':
            if node.attribute[0].name == 'layout' and node.attribute[0].s.lower() == 'bsnd':
                shape = [1, max_seq_len, num_kv_heads, dim]
            else:
                shape = [1, num_kv_heads, max_seq_len, dim]
            value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT16, shape)
            if node.output[0] in graph_output:
                output = graph_output[node.output[0]]
                output.type.tensor_type.shape.CopyFrom(
                    value_info.type.tensor_type.shape)
            else:
                graph.value_info.append(value_info)
        elif node.op_type in ['MsRmsNorm', 'MsAddRmsNorm']:
            shape = [1, chunk_size, num_q_heads *
                     dim] if is_prefill else [1, 1, num_q_heads * dim]
            for i in range(len(node.output)):
                value_info = helper.make_tensor_value_info(
                    node.output[i], TensorProto.FLOAT16, shape)
                graph.value_info.append(value_info)
        elif node.op_type == 'MsRotaryPosEmb':
            rope_q_shape = [1, num_q_heads, chunk_size,
                            dim] if is_prefill else [1, num_q_heads, 1, dim]
            rope_k_shape = [1, num_kv_heads, chunk_size,
                            dim] if is_prefill else [1, num_kv_heads, 1, dim]

            rope_q_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT16, rope_q_shape)
            rope_k_value_info = helper.make_tensor_value_info(
                node.output[1], TensorProto.FLOAT16, rope_k_shape)
            graph.value_info.append(rope_q_value_info)
            graph.value_info.append(rope_k_value_info)
        elif node.op_type == 'MsGroupMatmul':
            if node.attribute[0].name == 'trans_b' and node.attribute[0].s.lower() == 'true':
                shape = [1, num_q_heads, chunk_size, max_seq_len] if is_prefill else [
                    1, num_q_heads, 1, max_seq_len]
            else:
                shape = [1, num_q_heads, chunk_size, dim] if is_prefill else [
                    1, num_q_heads, 1, dim]
            value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT16, shape)
            graph.value_info.append(value_info)


def quantize_weight_g32_4bit_nd(weight):
    '''W4A16 weight quantize, 4bits, symmetric, group size=128'''
    def ceil(x, y):
        return (x + y - 1) // y * y

    # Transpose [K, N] to [N, K]
    K, N = weight.shape
    weight = weight.T
    weight = weight.astype(np.float32)

    # Caculate Q4_0 weight length
    group_size = 32
    frac_n = 16

    N_aligned = ceil(N, frac_n)
    length = N_aligned * (K // 2 + K // group_size * 2)
    quant_weight = (ctypes.c_int8 * length)()
    lib_q4_gp32.quantize_Q4_N_0_V1_reference(
        weight.tobytes(), quant_weight, N, K)
    return np.frombuffer(quant_weight, np.int8)


def quantize_weight_g128_4bit_nz(weight):
    '''W4A8 weight quantize, 4bits, symmetric, group size=128'''
    def ceil(x, y):
        return (x + y - 1) // y * y

    # Transpose [K, N] to [N, K]
    k, n = weight.shape
    weight = weight.T
    weight = weight.astype(np.float32)

    # Caculate Q4_0 weight length
    group_size = 128
    frac_n = 16

    n_aligned = ceil(n, frac_n)
    length = n_aligned * (k // 2 + k // group_size * 4)
    quant_weight = (ctypes.c_int8 * length)()
    lib_q4.quantize_row_q4_n_0_nz_reference(
        weight.tobytes(), quant_weight, n, k)
    return np.frombuffer(quant_weight, np.int8)


def get_shape_info(graph):
    '''get operator shape'''
    shape_info = {}
    for value_info in graph.value_info:
        tensor_type = value_info.type.tensor_type

        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param)

        if len(shape) != 0:  # pylint: disable=C1801
            shape_info[value_info.name] = shape
    return shape_info


def quant_node_4bit_gp32(shape_info, origin_node, initializers):
    '''W4A16 quantization'''
    new_node_list = []
    new_initializer_list = []
    weight_name = origin_node.input[1]
    weight_init = initializers[weight_name]
    weight_data = onnx.numpy_helper.to_array(weight_init)
    weight_shape = weight_data.shape
    quantized_weight = quantize_weight_g32_4bit_nd(weight_data)
    quant_weight_name = weight_name + "_quant"
    quant_weight_init = onnx.numpy_helper.from_array(
        quantized_weight, quant_weight_name)
    new_initializer_list.append(quant_weight_init)

    # Create activation node (int8)
    input_name = origin_node.input[0]
    # quant_act_name = input_name + "_quant"

    # Create Linear layer (A8W4)
    quant_linear_output = origin_node.output[0]
    if input_name not in shape_info:
        raise ValueError(f'Can not get input shape of {origin_node.name}.')

    act_shape = shape_info[input_name]
    quant_linear_node = helper.make_node(
        "MsQuant4N0Group32",
        inputs=[input_name, quant_weight_name],
        outputs=[quant_linear_output],
        input1_shape=weight_shape,
        name=origin_node.name + "_quant"
    )
    new_node_list.append(quant_linear_node)
    return new_node_list, new_initializer_list


def quant_node_4bit(shape_info, origin_node, initializers):
    '''W4A8 quantization'''
    new_node_list = []
    new_initializer_list = []
    weight_name = origin_node.input[1]
    weight_init = initializers[weight_name]
    weight_data = onnx.numpy_helper.to_array(weight_init)
    weight_shape = weight_data.shape
    quantized_weight = quantize_weight_g128_4bit_nz(weight_data)
    quant_weight_name = weight_name + "_quant"
    quant_weight_init = onnx.numpy_helper.from_array(
        quantized_weight, quant_weight_name)
    new_initializer_list.append(quant_weight_init)

    # Create activation node (int8)
    input_name = origin_node.input[0]
    quant_act_name = input_name + "_quant"
    quant_act_node = helper.make_node(
        "MsFloatCastInt",
        inputs=[input_name],
        outputs=[quant_act_name],
        name=quant_act_name
    )
    new_node_list.append(quant_act_node)

    # Create Linear layer (A8W4)
    quant_linear_output = origin_node.output[0]
    if input_name not in shape_info:
        raise ValueError(f'Can not get input shape of {origin_node.name}.')

    quant_linear_node = helper.make_node(
        "MsQuant4N0Group128",
        inputs=[quant_act_name, quant_weight_name],
        outputs=[quant_linear_output],
        input1_shape=weight_shape,
        name=origin_node.name + "_quant"
    )
    new_node_list.append(quant_linear_node)
    return new_node_list, new_initializer_list


def quant_node_2bit(shape_info, origin_node, initializers, q2_v):
    '''W2A16 quantization'''
    new_node_list = []
    new_initializer_list = []
    # Get fp16 weight
    weight_name = origin_node.input[1]
    weight_init = initializers[weight_name]
    weight_data = onnx.numpy_helper.to_array(weight_init)
    weight_shape = weight_data.shape

    # Quant weight to 2bit
    quantized_weight = quantize_weight_g32_2bit_nd(weight_data)
    quant_weight_name = weight_name + "_quant"
    quant_weight_init = onnx.numpy_helper.from_array(
        np.frombuffer(quantized_weight.tobytes() +
                      q2_v.tobytes(), np.uint8), quant_weight_name
    )
    new_initializer_list.append(quant_weight_init)

    # # Create activation node (int8)
    input_name = origin_node.input[0]

    # Create Linear layer (A16W2)
    quant_linear_output = origin_node.output[0]
    if input_name not in shape_info:
        raise ValueError(f"Can not get input shape of {node.name}.")

    quant_linear_node = helper.make_node(
        "MsQuant2N0Group32",
        inputs=[origin_node.input[0], quant_weight_name],
        outputs=[quant_linear_output],
        input1_shape=weight_shape,
        name=origin_node.name + "_quant",
    )
    new_node_list.append(quant_linear_node)
    return new_node_list, new_initializer_list


def quantize_linear_ops(model, embedding_quant_config, decoder_quant_config):
    '''decoder Linear quantization'''
    graph = model.graph

    # Get operator shapes
    shape_info = get_shape_info(graph)

    # Keep quant weights and nodes
    new_initializers = []
    new_nodes = []

    # Find all Linear operators
    linear_nodes = []
    lmhead_node = None
    for node in graph.node:
        if node.op_type in ['MatMul']:
            if 'lm_head' in node.name:
                lmhead_node = node
                continue
            linear_nodes.append(node)

    initializers = {init.name: init for init in graph.initializer}
    # init q2_v_index
    if embedding_quant_config.quant_method == "W2A16" or decoder_quant_config.quant_method == "W2A16":
        q2_v = load_q2_constant()

    # lmhead quantize
    if embedding_quant_config.is_quant:
        if embedding_quant_config.quant_method == "W4A8":
            new_node_list, new_initializer_list = quant_node_4bit(
                shape_info, lmhead_node, initializers)
            new_nodes.extend(new_node_list)
            new_initializers.extend(new_initializer_list)
        elif embedding_quant_config.quant_method == "W2A16":
            new_node_list, new_initializer_list = quant_node_2bit(
                shape_info, lmhead_node, initializers, q2_v)
            new_nodes.extend(new_node_list)
            new_initializers.extend(new_initializer_list)
        elif embedding_quant_config.quant_method == "W4A16":
            new_node_list, new_initializer_list = quant_node_4bit_gp32(
                shape_info, lmhead_node, initializers)
            new_nodes.extend(new_node_list)
            new_initializers.extend(new_initializer_list)
        else:
            raise RuntimeError(
                f"quant method: {embedding_quant_config.quant_method} not supported")

    for node in linear_nodes:
        if node.input[1] not in initializers:
            # Save MatMul of self_attn for build graph
            new_nodes.append(node)
            continue
        if decoder_quant_config.quant_method == "W4A8":
            new_node_list, new_initializer_list = quant_node_4bit(
                shape_info, node, initializers)
            new_nodes.extend(new_node_list)
            new_initializers.extend(new_initializer_list)
        elif decoder_quant_config.quant_method == "W2A16":
            new_node_list, new_initializer_list = quant_node_2bit(
                shape_info, node, initializers, q2_v)
            new_nodes.extend(new_node_list)
            new_initializers.extend(new_initializer_list)
        elif decoder_quant_config.quant_method == "W4A16":
            new_node_list, new_initializer_list = quant_node_4bit_gp32(
                shape_info, node, initializers)
            new_nodes.extend(new_node_list)
            new_initializers.extend(new_initializer_list)
        else:
            raise RuntimeError(
                f"quant method: {decoder_quant_config.quant_method} not supported")

    # Add nodes except MatMul
    for node in graph.node:
        if node not in linear_nodes:
            if 'lm_head' in node.name and embedding_quant_config.is_quant:
                continue
            new_nodes.append(node)

    # Add constant node except Linear weight
    for ori_init in graph.initializer:
        is_quanted = False
        for new_init in new_initializers:
            if ori_init.name + '_quant' == new_init.name:
                is_quanted = True
                break

        if not is_quanted:
            new_initializers.append(ori_init)

    # Create new graph
    new_graph = helper.make_graph(
        new_nodes,
        graph.name,
        graph.input,
        graph.output,
        new_initializers
    )

    # Creat new model
    new_model = helper.make_model(new_graph, producer_name=model.producer_name)
    new_model.opset_import[0].version = 18

    apply_shared_weight(new_model, is_quant=embedding_quant_config.is_quant)

    # Remove redundant MsFloatCastInt node
    new_model = slim(new_model)

    # Duplicate shared constant node
    duplicate_shared_initializers(new_model)
    return new_model


def quantize_weight_g32_2bit_nd(weight):
    '''W2A16 weight quantization'''
    def ceil(x, y):
        return (x + y - 1) // y * y

    # Transpose [K, N] to [N, K]
    k, n = weight.shape
    weight = weight.T
    weight = weight.astype(np.float32)

    # Caculate Q4_0 weight length
    group_size = 32
    frac_n = 16

    n_aligned = ceil(n, frac_n)
    length = n_aligned * (k // 4 + k // group_size * 2)
    quant_weight = (ctypes.c_uint8 * length)()
    lib_q2.quantize_Q2_N_0_reference(weight.tobytes(), quant_weight, n, k)
    return np.frombuffer(quant_weight, np.uint8)


def load_q2_constant():
    q2_v = np.array([
        0,  16,   1,  17,   2,  18,   3,  19,   4,  20,   5,  21,   6,  22,   7,  23,
        32,  48,  33,  49,  34,  50,  35,  51,  36,  52,  37,  53,  38,  54,  39,  55,
        64,  80,  65,  81,  66,  82,  67,  83,  68,  84,  69,  85,  70,  86,  71,  87,
        96, 112,  97, 113,  98, 114,  99, 115, 100, 116, 101, 117, 102, 118, 103, 119,
        8,  24,   9,  25,  10,  26,  11,  27,  12,  28,  13,  29,  14,  30,  15,  31,
        40,  56,  41,  57,  42,  58,  43,  59,  44,  60,  45,  61,  46,  62,  47,  63,
        72,  88,  73,  89,  74,  90,  75,  91,  76,  92,  77,  93,  78,  94,  79,  95,
        104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127
    ], dtype=np.uint8)
    return q2_v


def infer_shape(model, chunk_size, max_seq_len, num_kv_heads, num_q_heads, dim, is_prefill=True):
    custom_op_infer_shape(model.graph, chunk_size, max_seq_len,
                          num_kv_heads, num_q_heads, dim, is_prefill)
    model = shape_inference.infer_shapes(model)
    return model


def apply_shared_weight(model, is_quant=False):
    '''Apply WordEmbedding & LmHead weight shared'''
    graph = model.graph
    for node in graph.node:
        if '/lm_head/MatMul' not in node.name:
            continue

        weight_name = node.input[1]
        for init in graph.initializer:
            if init.name != weight_name:
                continue

            weight_data = onnx.numpy_helper.to_array(init)

            if is_quant:
                embedding_input = helper.make_tensor_value_info(
                    'embedding_weight', TensorProto.INT8, weight_data.shape)
            else:
                embedding_input = helper.make_tensor_value_info('embedding_weight', TensorProto.FLOAT16,
                                                                weight_data.T.shape)

            node_input = 'embedding_weight'
            if not is_quant:
                transpose_node = helper.make_node(
                    "Transpose",
                    inputs=['embedding_weight'],
                    outputs=['embedding_weight_transpose'],
                    perm=[1, 0],
                    name='embedding_weight_transpose'
                )
                node_input = 'embedding_weight_transpose'
                model.graph.node.append(transpose_node)

            node.input[1] = node_input
            model.graph.input.insert(6, embedding_input)
            model.graph.initializer.remove(init)
            print("apply lm head weight shared.")
            return
    return


def apply_quant(input_model, output_model, model_config: ModelConfig):
    '''model quantization'''
    model = onnx.load(input_model)

    print("Start infer shape...")
    # Ensure that the Linear shape can be obtained before model quantization.
    model = infer_shape(model,
                        model_config.chunk_size,
                        model_config.max_length,
                        model_config.num_attention_heads,
                        model_config.num_key_value_heads,
                        model_config.hidden_size // model_config.num_key_value_heads)
    print("Infer shape completed...")

    print("Start model quantization...")
    model = quantize_linear_ops(
        model, model_config.embedding_quant, model_config.decoder_quant)
    print("Model quantize finished!")

    # save model
    onnx.save(model, output_model)
    print(f"Save quant model to: {output_model}")
