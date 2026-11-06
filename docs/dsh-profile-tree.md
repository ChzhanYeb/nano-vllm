# W1-T5 dsh Profile 插件树与 Patch 加载机制

## 1. headless profile 插件树
完整输出见 `docs/dsh-profile-tree-headless.txt`。
关键挂载点：`id: llm, name: '@deepseek-ai/dsh-llm'` → T6 在此注册 adapter。

## 2. Patch 三层写法
### 2.1 bundle 层（插件包内 cordis.patch.yml）
- insert: - id: provider-nanovllm, name: provider-nanovllm

### 2.2 用户层（~/.dsh/profiles/eval/cordis.patch.yml）
由 `dsh plugin add` 自动追加，内容与 bundle 层一致。

### 2.3 覆盖层（--patch <abs-file>）
路径必须绝对，优先级最高，适合 T6 源码级调试。

## 3. 加载验证记录
命令：dsh --profile eval --dump-config | grep -A 2 provider-nanovllm
输出：（贴入实际输出）

## 4. T6 挂载点
ctx.llm.registerAdapter(['nanovllm'], adapter)