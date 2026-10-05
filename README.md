# 航摄姿态插值服务（Attitude Interpolation Service）

将低频惯导姿态按最短旋转弧（slerp）投影到相机曝光时刻。服务在插值前先把四元数
的 ±等价表示统一成符号连续的链，从根本上避免相邻帧足迹翻转。

## 接口

### `POST /api/attitudes/interpolate`

请求体：

```json
{
  "samples": [
    {"t_ns": 1000000000, "q": [1.0, 0.0, 0.0, 0.0]},
    {"t_ns": 1010000000, "q": [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]}
  ],
  "queries": [1000000000, 1005000000, 1010000000],
  "max_interval_ns": 10000000
}
```

- `samples`：2–2000 个姿态样本，`t_ns` 为纳秒时间戳，**严格递增**；
  `q` 固定为 `[w, x, y, z]`，各分量必须为有限数且整体非零（服务按单位四元数解释，自动归一化）。
- `queries`：1–500 个查询时刻，严格递增；允许等于样本端点，但不得超出样本时间范围。
- `max_interval_ns`：允许的最大采样间隔（纳秒，正数）。每个相邻样本间隔都不得超过它。

响应（按查询顺序返回单位四元数）：

```json
{"quaternions": [[1.0, 0.0, 0.0, 0.0], ["..."], ["..."]]}
```

输出约定：

- 每个结果均为单位四元数，数值误差 ≤ 1e-9；
- 首项的第一个非零分量为正；
- 后续项选择与前一项内积非负的等价符号，保证跨帧连续。

### 关键规则

- **最短弧**：相邻旋转的夹角必须小于 180°（单位四元数内积绝对值需 > 1e-12）。
  180° 时最短弧不唯一，返回 `ambiguous_rotation`（带样本索引）。
- **无部分结果**：请求非法时返回 `422`（非法 JSON 为 `400`），所有可定位的问题
  一次性在 `details` 中给出（含 `type` / `loc` / `index` / `message`），响应体不含
  任何插值结果。错误类型包括：`zero_quaternion`、`non_finite`、
  `non_increasing_time`、`gap_too_large`、`out_of_range`、`ambiguous_rotation`、
  `out_of_bounds` 等。
- **纳秒精度**：整数量时间戳按 Python 任意精度整数比较与求差；在 1.7e18（epoch 纳秒）
  量级也不会因 float64 舍入误判间隔。

### 健康检查

`GET /health` → `200 {"status":"ok"}`

## 运行

宿主机端口通过 `HOST_PORT` 配置（默认 8080，容器内固定 8000）：

```bash
HOST_PORT=9090 docker compose up --build
```

`web` 服务通过健康检查后，一次性服务 `verify` 才会启动；它依次执行：

1. 构建检查（全量字节码编译）；
2. 代码测试（pytest 单元 + API 套件）；
3. API 冒烟：健康检查、合法插值（单位性/精度/符号连续）、超长间隙拒绝（索引可定位且无部分结果）。

`verify` 自行退出，退出码汇总全部结果（全过为 0，否则为 1）：

```bash
docker compose build
docker compose up web -d
docker compose run --rm verify    # 或随 `docker compose up` 自动执行
```

仅本地开发：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest
```

## 布局

```
app/quaternion.py      # 四元数运算与 slerp
app/interpolation.py   # 校验、符号连续化、插值流水线
app/main.py            # FastAPI 路由与错误响应
scripts/verify.py      # 一次性校验（build + pytest + 冒烟），退出码汇总
tests/                 # 单元与 API 测试
Dockerfile, docker-compose.yml
```
