Rosseland_opa_new CPU+GPU 异构版运行方法

编译：
cd /home/user/Rosseland_opa_new_hetero/equally
make clean
make

推荐最快运行命令（两张 GPU，关闭 CPU worker 调度开销）：
cd /home/user/Rosseland_opa_new_hetero/equally && make clean && make && taskset -c 0-$(($(nproc --all)-1)) env CUDA_VISIBLE_DEVICES=0,1 CUBACORES=0 ROSS_USE_GPU=1 ROSS_CUBAACCELS=2 ROSS_CUBAACCELMAX=1000000 CUBAVERBOSE=1 /usr/bin/time -f 'elapsed=%e user=%U sys=%S cpu=%P' ./SNOP_op_Ross3

CPU+GPU 都参与的运行命令（比最快命令稍慢）：
cd /home/user/Rosseland_opa_new_hetero/equally && make clean && make && taskset -c 0-$(($(nproc --all)-1)) env CUDA_VISIBLE_DEVICES=0,1 ROSS_USE_GPU=1 ROSS_CUBAACCELS=2 ROSS_CUBAACCELMAX=950000 CUBAVERBOSE=1 /usr/bin/time -f 'elapsed=%e user=%U sys=%S cpu=%P' ./SNOP_op_Ross3

说明：
1. CUDA_VISIBLE_DEVICES=0,1 表示使用两张 RTX 5090。
2. CUBACORES=0 表示最快模式下不启动 CPU worker，避免 GPU 已吃满批次时 CPU 调度拖慢。
3. ROSS_CUBAACCELS=2 表示启用两个 Cuba accelerator。
4. ROSS_CUBAACCELMAX=1000000 表示每张 GPU 每批处理 1000000 个采样点。
5. ROSS_USE_GPU=0 可关闭 GPU，只用 CPU 路径做对照。

输出：
data.txt 每组参数追加一行：
rod  tep  tgama  lnu

测试结果：
前三组参数对照中，原版 data.txt 和异构版 data.txt 字节级一致。
原版前三组耗时 60.79 s。
单 GPU+CPU 异构前三组耗时 13.22 s，加速约 4.60 倍。
单独第一组测试中，原版 21.89 s，单 GPU+CPU 异构版 4.34 s，加速约 5.04 倍。
双 GPU 最快模式第一组耗时 2.55 s，加速约 8.58 倍。
