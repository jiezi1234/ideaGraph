# 科研工作台部署

Web 入口复用既有建库与 chat-idea 流程，不通过子进程执行 CLI。单 worker 配合 SQLite 持久化任务，最多四个未完成任务，串行运行以限制内存与 API 并发。重启时未完成任务标记失败，由用户决定重试。

## 配置

复制 `.env.example` 为 `.env`，填写服务端模型、embedding、Neo4j 配置，并设置至少 24 字符的随机 `WEB_ACCESS_TOKEN`。该口令仅用于工作台登录，不是模型 API Key。当前工作台面向共享同一文献库的可信团队，不提供多租户隔离。

容器内使用 `NEO4J_URI=bolt://neo4j:7687`、`NEO4J_DATABASE=neo4j`。Neo4j 密码必须与 compose 环境一致。配置可下载的 tokenizer 路径，或者预置本地 tokenizer 文件。首次使用需要建立文献库。

```bash
docker compose --env-file .env -f deploy/compose.yml up -d --build
docker compose --env-file .env -f deploy/compose.yml exec web python3 -m pytest -q
```

Dockerfile 使用 CPU 版本 PyTorch，禁用 hybrid 额外依赖。2 核 4GB 推荐摘要建库、15 篇文献上限、抽取与 embedding 并发均为 1。全文解析应在资源充足后单独配置。

Compose 的 `proxy` 网络为本次服务器已有的 `backend_default`。其他服务器应替换该网络，或移除 proxy 网络并使用宿主机反向代理转发到 `127.0.0.1:8088`。

Neo4j 的 `7474`（Browser）和 `7687`（Bolt）在 compose 中映射到宿主机；如需公网访问，还要在云安全组同时放行这两个端口，并用 `NEO4J_server_default__advertised__address` 指定公网地址，否则 `neo4j://` 路由表会返回容器内地址导致外部客户端连不上。不需要公网访问时，把端口改回 `127.0.0.1:7474:7474` 即可。

## 页面与接口

- `/` 展示研究流程，连接工作台后可从最近任务恢复进度或打开报告。
- `/build` 接受种子检索关键词，完成后展示报告和共享图谱预览，可将主题带入研究构思。
- `/graph` 提供只读图谱浏览、节点搜索与摘要或描述摘录，可选择有摘要的论文作为研究起点。
- `/idea` 接受研究问题、起点论文和资源约束，展示问题、方法、实验计划及已启用的评审结果。
- 建图或构思页可通过 `?task=<id>` 恢复指定任务，展开「执行过程与调用详情」查看过程图和节点输入输出。
- `POST /api/tasks`、`GET /api/tasks/{id}`：任务提交与查询（Bearer 口令）
- `GET /api/graph`：返回论文与实体节点、关系边，供前端渲染（Bearer 口令）
- `GET /api/tasks/{id}/report.md`：下载该任务的 Markdown 报告（Bearer 口令）
- `GET /api/tasks/{id}/trace`：读取该任务的 Langfuse 调用摘要（Bearer 口令），服务端缓存 15 秒。
- `GET /api/tasks/{id}/events/{event_id}`：按需读取本地节点 input/output。
- `GET /api/tasks/{id}/trace/{observation_id}`：按任务绑定的 trace 和调用 ID 读取 Langfuse input/output。两个详情接口均要求 Bearer 口令。

### 任务过程与 Langfuse

建图和研究任务显示可缩放、拖拽、点击的过程图，包括模型准备、论文发现、实体抽取、图谱写入、检索、探索轮次和评审。连线表达真实父子调用 / 包含关系，不表达串行执行依赖；并行调用呈现为同一父节点下的分支。可在「实时进度」和「Langfuse 调用」之间切换，也可用下拉框选择节点。阶段进度保存在同一个 SQLite 文件中；任务接口返回最近 200 条记录，历史任务可以回看。页面约每 3.5 秒刷新状态，约每 17.5 秒同步远端详情。

配置 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`，并启用 `LANGFUSE_TRACING_ENABLED` 后，新 Web 任务会建立独立根 trace。选择节点可查看调用名称、耗时、token 用量及 Input / Output。详情采用 Observations API v2（`/api/public/v2/observations`），服务端需支持此接口；旧版自托管服务不支持时仅显示本地进度。每次概览读取最多 5 页、每页 1000 条，图中最多显示 500 个调用节点；截断时页面提示，缺失父节点显示灰色占位。

本地阶段状态不等待 Langfuse 入库。远端未配置、暂不可用或尚未入库时，任务继续执行并显示相应提示；完成后会补拉一次，也可手动刷新。输入输出在点击节点时加载，以纯文本 / JSON 展示；新模型调用记录 messages、工具定义和实际回答 / 工具调用，工作流记录任务输入和结果，建图里程碑记录论文标识及计数。凭据字段进行过滤，每项超过 100,000 字符会明确标记截断；异常原文不作为输出。旧记录缺少父关系或 io 时不会推断或补造数据，界面提示未记录。已有任务不会自动补建 trace。共享访问口令仍代表同一个可信工作空间，有口令的成员可查看这些输入输出，并非用户级权限隔离。

报告同时在 `exports/reports/{task_id}.md` 落盘，结果里也保留 `report_markdown` 字段。前端用本地 vendored 的 marked + DOMPurify 渲染，避免依赖外部 CDN。

## HTTPS 子路径

已有 Caddy 站点可追加下列路由，保留原服务的兜底路由：

```caddyfile
redir /research /research/
handle_path /research/* {
    reverse_proxy ideagraph-web-1:8080
}
```

前端使用相对资源路径；API 地址以当前页面路径为前缀。Web 服务保持一个 worker，不能用多 worker 或滚动多副本部署共享当前任务队列。

## 数据与运维

任务记录：`exports/web/tasks.sqlite3`；Markdown 报告：`exports/reports/`；科研归档：`exports/chat_idea_runs/`；日志：`logs/`；向量缓存：`.cache/`。Neo4j 使用命名 volume。备份时需同时保留数据库与上述目录。访问口令只保存在浏览器 sessionStorage，关闭会话后需重新输入。

`/health` 仅表示 Web 进程存活，不代表模型端点、OpenAlex 或文献库已就绪。失败详情留在服务器日志中，前端只返回不含凭据的提示。
