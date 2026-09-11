# 服务器提交与推送约定

2026-09-11 用户授权：后续与本项目任务相关、已经验证的改动，可直接 commit 与 push，无须逐次请示；不添加 Co-authored-by 或其他共同作者 trailer。

- 提交身份：`zyx-thu <yx-zheneg22@mails.tsinghua.edu.cn>`，仅设在本仓库。
- 用户明确限定服务器上的 GitHub 配置作用域为 `/home/zhengyuxi`：不写 Git system/global 配置，不改其他用户配置、共享 SSH 配置或系统程序目录。作者与认证 helper 仅写当前仓库 `.git/config`。
- GitHub CLI 位于 `/home/zhengyuxi/.local/bin/gh`；配置使用独立 `GH_CONFIG_DIR=/home/zhengyuxi/.config/gh-sam3`，每次命令与本仓库 helper 都显式传入，不向 shell 启动文件导出。令牌优先使用本用户的加密凭据库 `/home/zhengyuxi/.local/share/keyrings`，不强制明文存储；已核验配置/凭据目录权限 `700`、现有 login keyring 权限 `600`。若凭据库不可用，先停止认证并汇报，不擅自切换明文方案。
- 不使用会全局安装 helper 的 `gh auth setup-git`，非交互浏览器登录不自动配置 Git helper；不输出或提交访问令牌。此为本机用户/仓库隔离，不改变 GitHub 账号权限，也不对服务器管理员或同一 Linux 账号内的进程构成隔离。
- 当前工作分支：`sam3-learnable-tokens`。继续在该实验分支交付，不自动合并主分支、不强制推送或改写远端历史。
- 每次先检查差异和现有暂存区，保留用户无关改动；运行与改动风险相称的验证，并更新 CHANGELOG。
- 仅提交项目代码、测试和必要仓库说明；文档及复核媒体位于同级 `../docs`，运行产物位于忽略的 `runs/`，不上传数据、checkpoint、密钥或访问令牌。
- 登录账号必须对推送目标有写权限。首次登录、账号所有权验证及实际缺失的远端权限仍需要用户完成；不会把作者姓名当成已登录账号。
- 推送后核对远端分支指向。遇到远端新增提交、冲突或权限拒绝，保留本地提交，不强推、不自行更换到陌生仓库。
- 这是一条后续任务执行约定，不是运行在后台、会把任意未审核文件自动上传的守护进程。
