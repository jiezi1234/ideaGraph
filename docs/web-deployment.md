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

Compose 的 `proxy` 网络为本次服务器已有的 `backend_default`。其他服务器应替换该网络，或移除 proxy 网络并使用宿主机反向代理转发到 `127.0.0.1:8088`。数据库不暴露公网端口。

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

任务记录：`exports/web/tasks.sqlite3`；科研归档：`exports/chat_idea_runs/`；日志：`logs/`；向量缓存：`.cache/`。Neo4j 使用命名 volume。备份时需同时保留数据库与上述目录。访问口令只保存在浏览器 sessionStorage，关闭会话后需重新输入。

`/health` 仅表示 Web 进程存活，不代表模型端点、OpenAlex 或文献库已就绪。失败详情留在服务器日志中，前端只返回不含凭据的提示。
