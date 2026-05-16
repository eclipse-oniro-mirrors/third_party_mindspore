export MSLITE_PACKAGE_PATH=/home/c00619384/yangyc/code/tools/mindspore-lite-2.4.0-linux-x64 # 此处改成实际的MindSpore绝对路径！
export LD_LIBRARY_PATH=${MSLITE_PACKAGE_PATH}/runtime/lib:${MSLITE_PACKAGE_PATH}/tools/converter/lib:${MSLITE_PACKAGE_PATH}/runtime/third_party/dnnl/:${MSLITE_PACKAGE_PATH}
export PYTHONPATH=/home/c00619384/yangyc/code/MSLite_repo/mindspore/mindspore/lite/test/st/ops/frame:$PYTHONPATH

# 执行单算子用例
cd op
python run.py -a -o ../output