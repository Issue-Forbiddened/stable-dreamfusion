#!/bin/bash

# 设置目标压缩文件的名称
output_tar="compressed_runs.tar"

# 判断压缩文件是否已存在，如果存在则删除
if [ -f "$output_tar" ]; then
    rm "$output_tar"
fi

# 使用 find 命令查找名为 'run' 的文件夹，它们的父文件夹名以 'trial' 开头
find . -type d -name "run" -path "./trial*" | while read run_folder; do
    # 使用 tar 命令添加这些文件夹到压缩文件中
    tar --append --file="$output_tar" -C "." "$run_folder"
done

# 为了减小文件大小，对压缩文件进行gzip压缩
gzip "$output_tar"

echo "Finished compressing to ${output_tar}.gz"
