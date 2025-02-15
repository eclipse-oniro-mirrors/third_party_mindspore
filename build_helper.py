#!/usr/bin/env python3

import os
import shutil
import zipfile
import argparse

def extract_source(in_zip_path, out_src_path):
    "depress source code form release package"
    print('Extracting zipped release package...')
    f = zipfile.ZipFile(in_zip_path, "r")
    f.extractall(path=out_src_path)
    old_src_dir = out_src_path + "/mindspore-v2.1.0/"
    new_src_dir = out_src_path + "/source/"
    os.rename(old_src_dir, new_src_dir)
    print("Done extraction.")

def do_patch(patch_dir, target_dir):
    patches = [
        '0001-build-gn-c-api-for-OHOS.patch',
        '0002-train-and-build.patch',
        '0003-add-js-api.patch',
        '0004-cross-compile-ndkso-fp16-nnrt-train_capi.patch',
        '0005-micro-for-ohos.patch',
        '0006-remove-lite-expression-fix-double-loadso.patch',
        '0007-deobfuscator.patch',
        '0008-upgrade-flatbuffers-fix_crash.patch',
        '0009-npu-zero-copy.patch',
        '0010-micro-dynamic-shape-support-discrete-value.patch',
        '0011-fix-npu-infer-memory-leak-delete-liteGraph.patch',
        '0012-fix-int8-bug.patch',
    ]

    cwd = os.getcwd()
    os.chdir(target_dir)
    print('Change dir to', os.getcwd())
    os.system('git init .')
    os.system('git add .; git commit -m "init"')
    
    for patch in patches:
        print('Applying ', patch, '...')
        ret = os.system('git apply ' + patch_dir + '/' + patch)
        if ret != 0:
            raise Exception("Apply patch {} failed.".format(patch))
        os.system('git add .; git commit -m "auto-apply ' + patch + '"')
        print('Done')
    os.chdir(cwd)

def create_status_file(out_src_path):
    f = open(out_src_path + '/.status', 'w')
    f.write('ok')
    f.close


def main_work():
    parser = argparse.ArgumentParser(description="mindspore build helper")
    parser.add_argument('--in_zip_path')
    parser.add_argument('--out_src_path')
    parser.add_argument('--patch_dir')
    args = vars(parser.parse_args())

    in_zip_path = os.path.realpath(args['in_zip_path'])
    out_src_path = args['out_src_path']
    patch_dir = os.path.realpath(args['patch_dir'])

    if os.path.exists(out_src_path):
        shutil.rmtree(out_src_path)

    os.mkdir(out_src_path)
    out_src_path = os.path.realpath(out_src_path)

    extract_source(in_zip_path, out_src_path)

    do_patch(patch_dir, out_src_path + '/source/')

    create_status_file(out_src_path)


if __name__ == "__main__":
    main_work()

