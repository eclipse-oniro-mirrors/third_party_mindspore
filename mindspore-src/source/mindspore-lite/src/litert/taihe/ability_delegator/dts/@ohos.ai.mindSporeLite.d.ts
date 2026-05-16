/*	
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 * @kit MindSporeLiteKit
 */

import { Callback } from './@ohos.base';

/**
 * @namespace mindSporeLite
 * @syscap SystemCapability.AI.MindSporeLite
 * @stagemodelonly
 * @since 10 dynamic
 * @since 22 static
 */
declare namespace mindSporeLite {

  /**
   * Create a Model instance from file path
   * @param { string } model - model indicates model path to be loaded
   * @param { Context } context - context indicates model context information
   * @returns { Promise<Model> } the promise returned by the function.
   * @throws { BusinessError} 1000000 - Model path error. Possible causes: 1. The model path is null; 2. The model path does not exist.
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000002 - Failed to create native model. Possible causes: 1. Insufficient permission to access the model path; 2. The model file is corrupted.
   * @throws { BusinessError} 1000003 - Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromFile(
    model: string,
    context?: Context): Promise<Model>;

  /**
   * Create a Model instance from file path.
   * @param { string } model - model indicates model path to be loaded
   * @param { Callback<Model> } callback - the callback of model
   * @throws { BusinessError} 1000000 - Model path error. Possible causes: 1. The model path is null; 2. The model path does not exist.
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000002 - Failed to create native model. Possible causes: 1. Insufficient permission to access the model path; 2. The model file is corrupted.
   * @throws { BusinessError} 1000003 - Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromFile(
    model: string, callback: Callback<Model>): void;

  /**
   * Create a Model instance from file path.
   * @param { string } model - model indicates model path to be loaded
   * @param { Context } context - context indicates model context information
   * @param { Callback<Model> } callback - the callback of model
   * @throws { BusinessError} 1000000 - Model path error. Possible causes: 1. The model path is null; 2. The model path does not exist.
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000002 - Failed to create native model. Possible causes: 1. Insufficient permission to access the model path; 2. The model file is corrupted.
   * @throws { BusinessError} 1000003 - Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromFile(
    model: string,
    context: Context, callback: Callback<Model>): void;

  /**
   * Create a Model instance from buffer
   * @param { ArrayBuffer } model - model indicates model buffer to be loaded
   * @param { Context } [context] - context indicates model context information
   * @returns { Promise<Model> } the promise returned by the function.
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000003 - Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd.
   * @throws { BusinessError} 1000004 - Model buffer error. Possible causes: 1. The buffer size is 0; 2. The buffer is a null pointer.
   * @throws { BusinessError} 1000005 - Failed to create native model from buffer. Possible causes: 1. The buffer size is incorrect; 2. The buffer file is damaged.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromBuffer(
    model: ArrayBuffer,
    context?: Context): Promise<Model>;

  /**
   * Create a Model instance from buffer
   * @param { ArrayBuffer } model - model indicates model buffer to be loaded
   * @param { Callback<Model> } callback - the callback of model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000003 - Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd.
   * @throws { BusinessError} 1000004 - Model buffer error. Possible causes: 1. The buffer size is 0; 2. The buffer is a null pointer.
   * @throws { BusinessError} 1000005 - Failed to create native model from buffer. Possible causes: 1. The buffer size is incorrect; 2. The buffer file is damaged.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromBuffer(
    model: ArrayBuffer, callback: Callback<Model>): void;

  /**
   * Create a Model instance from buffer
   * @param { ArrayBuffer } model - model indicates model buffer to be loaded
   * @param { Context } context - context indicates model context information
   * @param { Callback<Model> } callback - the callback of model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000003 - Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd.
   * @throws { BusinessError} 1000004 - Model buffer error. Possible causes: 1. The buffer size is 0; 2. The buffer is a null pointer.
   * @throws { BusinessError} 1000005 - Failed to create native model from buffer. Possible causes: 1. The buffer size is incorrect; 2. The buffer file is damaged.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromBuffer(
    model: ArrayBuffer,
    context: Context, callback: Callback<Model>): void;

    /**
   * Creates a Model instance file description
   * @param { int } model - model indicates model file description to be loaded
   * @param { Context } [context] - context indicates model context information
   * @returns { Promise<Model> } the promise returned by the function.
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000007 - Failed to create native model from file descriptor (fd). Possible causes: 1. The file descriptor (fd) is incorrect; 2. The model file is damaged.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromFd(
    model: int,
    context?: Context): Promise<Model>;

  /**
   * Create a Model instance from file description
   * @param { int } model - model indicates model file description to be loaded
   * @param { Callback<Model> } callback - the callback of model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000007 - Failed to create native model from file descriptor (fd). Possible causes: 1. The file descriptor (fd) is incorrect; 2. The model file is damaged.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromFd(
    model: int, callback: Callback<Model>): void;

  /**
   * Create a Model instance from file description
   * @param { int } model - model indicates model file description to be loaded
   * @param { Context } context - context indicates model context information
   * @param { Callback<Model> } callback - the callback of model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000007 - Failed to create native model from file descriptor (fd). Possible causes: 1. The file descriptor (fd) is incorrect; 2. The model file is damaged.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  function loadModelFromFd(
    model: int,
    context: Context, callback: Callback<Model>): void;

  /**
   * Load train model from file
   * @param { string } model - model file path
   * @param { TrainCfg } [trainCfg] - model train configuration
   * @param { Context } [context] - model build context
   * @returns { Promise<Model> } the promise of the built model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000008 - Invalid model path in training. Possible causes: 1. The model path is null; 2. The model path does not exist.
   * @throws { BusinessError} 1000009 - Failed to create native training model from path. Possible causes: 1. The model file is incorrect; 2. The training configuration is incorrect.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  function loadTrainModelFromFile(
    model: string,
    trainCfg?: TrainCfg,
    context?: Context): Promise<Model>;

  /**
   * Load train model from buffer
   * @param { ArrayBuffer } model - model buffer
   * @param { TrainCfg } [trainCfg] - model train configuration
   * @param { Context } [context] - model build context
   * @returns { Promise<Model> } the promise of the built model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000010 - Invalid model buffer in training. Possible causes: 1. The model buffer size is incorrect; 2. The model buffer is null.
   * @throws { BusinessError} 1000011 - Failed to create native training model from buffer. Possible causes: 1. The model buffer is incorrect; 2. The training configuration is incorrect.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  function loadTrainModelFromBuffer(
    model: ArrayBuffer,
    trainCfg?: TrainCfg,
    context?: Context): Promise<Model>;

  /**
   * Load train model from file description
   * @param { int } model - model file description
   * @param { TrainCfg } [trainCfg] - model train configuration
   * @param { Context } [context] - model build context
   * @returns { Promise<Model> } the promise of the built model
   * @throws { BusinessError} 1000001 - Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect.
   * @throws { BusinessError} 1000012 - Failed to create native training model from file descriptor (fd). Possible causes: 1. The model file or file descriptor (fd) is incorrect; 2. The training configuration is incorrect.
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  function loadTrainModelFromFd(
    model: int,
    trainCfg?: TrainCfg,
    context?: Context): Promise<Model>;

  /**
   * Provides manages model function. Including get inputs, predict ,resize.
   * @typedef Model
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  interface Model {
    /**
     * The learning rate of the training model
     * @type {?double}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    learningRate?: double;

    /**
     * The running mode of the model
     * @type {?boolean}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    trainMode?: boolean;

    /**
     * Get model input tensors.
     * @returns { MSTensor[] } the MSTensor array of the inputs.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    getInputs(): MSTensor[];

    /**
     * Infer model
     * @param { MSTensor[] } inputs - indicates the MSTensor array of the inputs.
     * @param { Callback<MSTensor[]> }  callback - the callback of MSTensor array.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    predict(inputs: MSTensor[], callback: Callback<MSTensor[]>): void;

    /**
     * Infer model
     * @param { MSTensor[] } inputs - indicates the MSTensor array of the inputs.
     * @returns { Promise<MSTensor[]> } the promise returned by the function.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    predict(inputs: MSTensor[]): Promise<MSTensor[]>;

    /**
     * resize model input
     * @param { MSTensor[] } inputs - indicates the MSTensor array of the inputs.
     * @param { Array<Array<int>> } dims - indicates the target new shape array
     * @returns { boolean } the boolean result if the resize operation is successful
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    resize(inputs: MSTensor[], dims: Array<Array<int>>): boolean;

    /**
     * Train model by step
     * @param { MSTensor[] } inputs - indicates the MSTensor array of the inputs.
     * @returns { boolean } the boolean result if the runStep operation is successful
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    runStep(inputs: MSTensor[]): boolean;

    /**
     * Obtain all weights of the model
     * @returns { MSTensor[] } the weight tensors of the model
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    getWeights(): MSTensor[];

    /**
     * Update weights of the model
     * @param { MSTensor[] } weights - indicates the MSTensor array of the inputs
     * @returns { boolean } the boolean result if updating weights operation is successful
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    updateWeights(weights: MSTensor[]): boolean;

    /**
     * Setup training with virtual batches
     * @param { int } virtualBatchMultiplier - virtual batch multiplier, use any number < 1 to disable
     * @param { double } lr - learning rate to use for virtual batch, -1 for internal configuration
     * @param { double } momentum - batch norm momentum to use for virtual batch, -1 for internal configuration
     * @returns { boolean } the boolean result if the operation is successful
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    setupVirtualBatch(virtualBatchMultiplier: int, lr: double, momentum: double): boolean;

    /**
     * Export train model to file
     * @param { string } modelFile - model file path.
     * @param { QuantizationType } [quantizationType] - the quantization type, default NO_QUANT.
     * @param { boolean } [exportInferenceOnly] - whether to export a inference only model, default true.
     * @param { string[] } [outputTensorName] - the set of name of output tensor the exported inference model,
     * @returns { boolean } - the boolean result if the operation is successful
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    exportModel(
      modelFile: string,
      quantizationType?: QuantizationType,
      exportInferenceOnly?: boolean,
      outputTensorName?: string[]): boolean;

    /**
     * Export model's weights, which can be used in micro only. Only valid for Lite Train
     * @param { string } weightFile - weight file path
     * @param { boolean } [isInference] - whether to export weights from inference model, only support this is `true` for now, default true
     * @param { boolean } [enableFp16] - float-weight is whether to be saved in float16 format, default false
     * @param { string[] } [changeableWeightsName] - changeable weights name
     * @returns { boolean } the boolean result if the operation is successful
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    exportWeightsCollaborateWithMicro(
      weightFile: string,
      isInference?: boolean,
      enableFp16?: boolean,
      changeableWeightsName?: string[]): boolean;
  }

  /**
   * Enum for quantization type
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  export enum QuantizationType {
    /**
     * No quantization.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    NO_QUANT = 0,
    /**
     * Weight quantization.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    WEIGHT_QUANT = 1,
    /**
     * Full quantization.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    FULL_QUANT = 2,
  }
  /**
   * Enum for optimization level
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  export enum OptimizationLevel {
    /**
     * Do not change
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    O0 = 0,
    /**
     * Cast network to float16, keep batch norm and loss in float32
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    O2 = 2,
    /**
     * Cast network to float16, including batch norm
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    O3 = 3,
    /**
     * Choose optimization based on device
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    AUTO = 4,
  }

  /**
   * Provides the train configuration
   * @typedef TrainCfg
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  interface TrainCfg {
    /**
     * Array of loss name
     * @type {?string[]}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    lossName?: string[],
    /**
     * Train optimization level
     * @type {?OptimizationLevel}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    optimizationLevel?: OptimizationLevel,
  }

  /**
   * Provides the device configurations
   * @typedef Context
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  interface Context {
    /**
     * The target device
     * @type {?string[]}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    target?: string[];
    /**
     * The cpu device information
     * @type {?CpuDevice}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    cpu?: CpuDevice;
    /**
     * The NNRT device information
     * @type {?NNRTDevice}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    nnrt?: NNRTDevice;
  }

  /**
   * Provides the CPU device info
   * @typedef CpuDevice
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  interface CpuDevice {
    /**
     * The thread num
     * @type {?int}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    threadNum?: int;

    /**
     * The thread affinity mode
     * @type {?ThreadAffinityMode}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    threadAffinityMode?: ThreadAffinityMode;

    /**
     * The thread affinity core list
     * @type {?int[]}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    threadAffinityCoreList?: int[];

    /**
     * The precision mode
     * @type {?string}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    precisionMode?: string;
  }

  /**
   * Enum for performance mode
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  export enum PerformanceMode {
    /**
     * No performance mode preference
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PERFORMANCE_NONE = 0,
    /**
     * Low power consumption mode
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PERFORMANCE_LOW = 1,
    /**
     * Medium performance mode
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PERFORMANCE_MEDIUM = 2,
    /**
     * High performance mode
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PERFORMANCE_HIGH = 3,
    /**
     * Ultimate performance mode
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PERFORMANCE_EXTREME = 4,
  }

  /**
   * Enum for scheduling priority
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  export enum Priority {
    /**
     * No priority preference
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PRIORITY_NONE = 0,
    /**
     * Low priority
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PRIORITY_LOW = 1,
    /**
     * Medium priority
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PRIORITY_MEDIUM = 2,
    /**
     * High priority
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    PRIORITY_HIGH = 3,
  }

  /**
   * Provides the extension information of nnrt device
   * @typedef Extension
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  interface Extension {
    /**
     * Extension name
     * @type {string}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    name: string,
    /**
     * Extension array buffer
     * @type {ArrayBuffer}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    value: ArrayBuffer
  }

  /**
   * Enum for nnrt device type
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  export enum NNRTDeviceType {
    /**
     * Devices that are not CPU, GPU, or dedicated accelerator
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    NNRTDEVICE_OTHERS = 0,

    /**
     * CPU device
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    NNRTDEVICE_CPU = 1,

    /**
     * GPU device
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    NNRTDEVICE_GPU = 2,

    /**
     * Dedicated hardware accelerator
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    NNRTDEVICE_ACCELERATOR = 3,
  }

  /**
   * Provides the nnrt device description
   * @typedef NNRTDeviceDescription
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  interface NNRTDeviceDescription {
    /**
     * Get device id
     * @returns { bigint } the number of device id
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    deviceID() : bigint;
    /**
     * Get device type.
     * @returns { NNRTDeviceType } the device type
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    deviceType() : NNRTDeviceType;
    /**
     * Get device name.
     * @returns { string } device name
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    deviceName() : string;
  }

  /**
   * Obtain the all device descriptions in NNRT.
   * @returns { NNRTDeviceDescription[] } the array of NNRTDeviceDescription
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 12 dynamic
   * @since 22 static
   */
  function getAllNNRTDeviceDescriptions() : NNRTDeviceDescription[];

  /**
   * Provides the NNRT device info
   * @typedef NNRTDevice
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  interface NNRTDevice {
    /**
     * NNRT device id.
     * @type {?bigint}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    deviceID?: bigint,
    /**
     * NNRT device performance mode.
     * @type {?PerformanceMode}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    performanceMode?: PerformanceMode,
    /**
     * NNRT device priority.
     * @type {?Priority}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    priority?: Priority,
    /**
     * NNRT device extension array.
     * @type {?Extension[]}
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 12 dynamic
     * @since 22 static
     */
    extensions?: Extension[],
  }

  /**
   * Enum for provides CPU thread affinity mode
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  export enum ThreadAffinityMode {
    /**
     * Thread affinity mode is no bind.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NO_AFFINITIES = 0,

    /**
     * Thread affinity mode is big cores first
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    BIG_CORES_FIRST = 1,

    /**
     * Thread affinity mode is little cores first
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    LITTLE_CORES_FIRST = 2,
  }

  /**
   * Provides MSTensor definition
   * @typedef MSTensor
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  interface MSTensor {
    /**
      * The name of the tensor.
      * @type {string}
      * @syscap SystemCapability.AI.MindSporeLite
      * @stagemodelonly
      * @since 10 dynamic
      * @since 22 static
      */
    name: string;

    /**
      * The shape of the tensor.
      * @type {int[]}
      * @syscap SystemCapability.AI.MindSporeLite
      * @stagemodelonly
      * @since 10 dynamic
      * @since 22 static
      */
    shape: int[];

    /**
      * The data size of the tensor.
      * @type {int}
      * @syscap SystemCapability.AI.MindSporeLite
      * @stagemodelonly
      * @since 10 dynamic
      * @since 22 static
      */
    elementNum: int;

    /**
      * The data size of the tensor.
      * @type {int}
      * @syscap SystemCapability.AI.MindSporeLite
      * @stagemodelonly
      * @since 10 dynamic
      * @since 22 static
      */
    dataSize: int;

    /**
      * The data type of the tensor.
      * @type {DataType}
      * @syscap SystemCapability.AI.MindSporeLite
      * @stagemodelonly
      * @since 10 dynamic
      * @since 22 static
      */
    dtype: DataType;
    /**
      * The format of the tensor.
      * @type {Format}
      * @syscap SystemCapability.AI.MindSporeLite
      * @stagemodelonly
      * @since 10 dynamic
      * @since 22 static
      */
    format: Format;

    /**
     * Get MSTensor data
     * @returns { ArrayBuffer } the data of tensor
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    getData(): ArrayBuffer;

    /**
     * Set MSTensor data
     * @param { ArrayBuffer } inputArray - indicates the buffer of tensor
     * @throws { BusinessError} 1000013 - Failed to set MSTensor data. Possible causes: 1. The input array buffer size is incorrect.
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    setData(inputArray: ArrayBuffer): void;
  }

  /**
   * Enum for provides MSTensor data type
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */
  export enum DataType {
    /**
     * data type is unknown
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    TYPE_UNKNOWN = 0,
   /**
     * data type is int8
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_INT8 = 32,
   /**
     * data type is int16
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_INT16 = 33,
   /**
     * data type is int32
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_INT32 = 34,
   /**
     * data type is int64
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_INT64 = 35,
   /**
     * data type is uint8
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_UINT8 = 37,
   /**
     * data type is uint16
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_UINT16 = 38,
   /**
     * data type is uint32
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_UINT32 = 39,
   /**
     * data type is uint64
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_UINT64 = 40,
   /**
     * data type is float16
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_FLOAT16 = 42,
   /**
     * data type is float32
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_FLOAT32 = 43,
   /**
     * data type is float64
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NUMBER_TYPE_FLOAT64 = 44,
  }

  /**
   * Enum for provides MSTensor format
   * @enum {int}
   * @syscap SystemCapability.AI.MindSporeLite
   * @stagemodelonly
   * @since 10 dynamic
   * @since 22 static
   */ 
  export enum Format {
   /**
     * data format is default
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */  
    DEFAULT_FORMAT = -1,
   /**
     * data format is NCHW
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */  
    NCHW = 0,
   /**
     * data format is NHWC
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NHWC = 1,
   /**
     * data format is NHWC4
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    NHWC4 = 2,
   /**
     * data format is HWKC
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    HWKC = 3,
   /**
     * data format is HWCK
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    HWCK = 4,
   /**
     * data format is KCHW
     * @syscap SystemCapability.AI.MindSporeLite
     * @stagemodelonly
     * @since 10 dynamic
     * @since 22 static
     */
    KCHW = 5,
  }
}
export default mindSporeLite;
