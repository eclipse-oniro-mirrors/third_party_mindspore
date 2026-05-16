# 测试用例运行指导

本指导以rk3567为例，介绍MindSpore测试用例的执行流程

1. 编译测试用例

    调用以下命令编译MindSpore单元测试用例和系统测试用例。

    ```shell
    ./build.sh --product-name rk3568 --ccache --build-target mindspore_test_target --jobs 4
    ```

    编译完成后，在`out/rk3568/tests`目录下找到单元测试用例和系统测试用例，如下图所示：

    ```text
    /out/rk3568/tests
    ├── systemtest                         # 系统测试用例存放目录
    │   └── mindspore         # MindSpore系统测试用例存放目录
    └── unittest                           # 单元测试用例存放目录
        └── mindspore         # MindSpore测试单元用例存放目录
    └── fuzztest                           # Fuzz测试用例存放目录
              └── mindspore         # MindSpore Fuzz测试单元用例存放目录
    ```

2. 上传测试用例。

    执行以下代码，将测试用例推送到设备。

    ```shell
    hdc shell "mkdir /data/local/tmp/mindspore_test"
    hdc file send ./out/rk3568/tests/unittest/mindspore/. /data/local/tmp/mindspore_test
    hdc file send ./out/rk3568/tests/systemtest/mindspore/. /data/local/tmp/mindspore_test
    ```

3. 执行单元测试用例。

    以`MindSporeUnitTest`为例，执行单元测试。

    ```shell
    hdc shell "chmod 755 /data/local/tmp/mindspore_test/MindSporeUnitTest"
    hdc shell "/data/local/tmp/mindspore_test/MindSporeUnitTest"
    ```

    如果用例全部通过，应该得到以下输出：

    ```text
    [==========] 1 tests from 1 test suite ran. (101ms total)
    [  PASSED  ] 1 tests.
    ```

4. 执行系统测试用例（可选）。

    以`MindSporeSystemTest`为例，执行以下指令，运行系统测试。

    ```shell
    hdc shell "chmod 755 /data/local/tmp/mindspore_test/MindSporeUnitTest"
    hdc shell "/data/local/tmp/mindspore_test/End2EndTest"
    ```

    如果用例全部通过，应该得到以下输出：

    ```text
    [==========] 1 tests from 1 test suite ran. (648ms total)
    [  PASSED  ] 1 tests.
    ```
