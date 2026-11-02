# 2026-10-20 迁移包验证

## 2026-11-02 快照补充

- 当前源码校验清单更新为73个 `serve/` Python文件，对应本次迁移快照，不再把当前版本宣称为10月20日k63原始代码。
- 本机CPU检查：预约表/预约调度23项、profile扫描/Verify等待6项均通过；待提交Python与shell语法检查、Git空白检查通过。
- Git内附LCB128的样本ID、顺序与input IDs，与当前profile使用的旧日志payload逐项一致；迁移无需下载或复制旧日志才能得到同一批输入。
- 新增 `profile-migration_zh.md`，提供显式模型/输入路径的组件采集与Nsight包装命令；旧参考机硬件启动脚本中的绝对路径尚未整体改写，不在新机直接照搬。
- 本次不新启动GPU推理，也没有在第二台服务器从零安装验证。下方10月20日GPU短测与k63结果是历史证据，不自动代表更新后的预约调度代码。

## 已完成

- 当前68个 `serve/` Python 文件的 SHA256 与正在运行的k63冻结快照完全一致。
- 189个Python包的完整版本、CUDA/编译器conda显式列表、FlashInfer/RAFT/vLLM子模块提交已导出。
- 原生依赖归档的808个文件解包后逐文件 SHA256 一致；模型小文件归档也附整体及逐文件哈希。
- Python脚本编译、shell语法、`--prepare-only` 的模型/输入/参数检查通过。
- 原生u截获的15组CPU检查：返回张量不变，后续随机抽样和RNG状态逐位一致；异常后恢复原函数。
- u日志关联检查：混合请求分段、float64文件往返、首次拒绝重建通过；人为篡改u会被拒绝。
- 便携入口TP2实跑：GPU2/3、Qwen3-8B、k63、r=.05、Graph/CPU/KV异步开启；4条输入，batch上限4，输出上限256，CPU页8192。
- 上述4条均完成；每rank共46轮，其中4轮cold-start、42轮稀疏。两rank的u记录与实际接受记录均一致；用u和p/q重建首次拒绝点，两个rank均0个无法解释的差异、0个比较边界差异。前置选集与下一轮verify关联通过。
- `repro/check_environment.py --gpu` 的依赖版本/自定义FlashInfer/动态库检查通过，重新解包原生依赖后在GPU2成功执行TopK短测。
- 独立克隆目录中，不存在原 `log/acceptance-k63-rscan-20261020`，依然通过r=.05/.25的CPU参数准备、入口帮助、全部Python脚本编译和冻结源码哈希检查；从Git内附证据包解压后，无GPU复核两个rank各46轮u/p/q/接受长度也通过。

短测引擎窗口11.064秒，生成1069个token；有45个token超过合计1024的请求输出预算，这是现有引擎按verify轮终止的行为，保存在审计中，没有裁掉结果或改成严格256输出。这些短测计时有诊断开销，不作为正式吞吐对照。

`experiments/acceptance/reference/actual-u-smoke-evidence.tar.gz` 保存这次短测的实际u、p/q和逐轮接受记录，可解包到临时目录后运行 `python -m experiments.acceptance.verify_uniforms /path/to/unpacked`，无需GPU即可复核。内部payload中的旧绝对output路径属于历史记录，校验入口使用命令行指定的解包目录。

## 验证边界

本机短测复用了已安装的Python环境和JIT缓存，使用重新解包的原生库；没有在一台全新服务器重新下载并安装189个包。新机器需按README安装并通过环境自检和短测。尚无依据保证不同硬件的吞吐相同。

当前只记录原生verify阶段实际生成的u。提前抽u、按u选择k或动态调度策略均未实现，短测不构成其收益证据。原有完整k63扫描没有u，不能事后补出对应原始随机数。
