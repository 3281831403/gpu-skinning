"""GPU 蒙皮练习的核心实现。

数据约定
--------
- 4x4 矩阵使用长度为 16 的扁平 list，按**行主序**存放；平移分量位于
  ``m[3]``、``m[7]``、``m[11]``。
- 顶点是 ``{"pos": [x, y, z], "normal": [nx, ny, nz],
  "weights": [{"bone": 0, "w": 1.0}, ...]}``。
- 每个顶点最多 4 个非零骨骼影响，权重和必须为 1，数值容差 1e-6。
- 本模块只依赖标准库 ``math``，不联网、不使用随机数。
"""

import math


class SkinError(ValueError):
    """蒙皮数据、骨骼索引或矩阵非法时抛出的异常。"""


_DET_EPS = 1e-12
_SUM_EPS = 1e-6
_MAX_INFLUENCES = 4


# ---------------------------------------------------------------------------
# 内部校验与工具
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_vector(value, length: int, name: str) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SkinError(f"{name} must be a sequence of {length} numbers")
    if not all(_is_number(item) for item in value):
        raise SkinError(f"{name} must contain only numbers")
    return list(value)


def _require_matrix(matrix, name: str = "matrix") -> list:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 16:
        raise SkinError(f"{name} must be a flat 4x4 matrix of length 16")
    if not all(_is_number(item) for item in matrix):
        raise SkinError(f"{name} must contain only numbers")
    return list(matrix)


def _require_weights(weights, name: str = "weights") -> list[dict]:
    """校验权重结构并返回 ``[{"bone": int, "w": float}, ...]`` 的新 list。"""
    if not isinstance(weights, (list, tuple)):
        raise SkinError(f"{name} must be a list")
    parsed = []
    for entry in weights:
        if not isinstance(entry, dict) or "bone" not in entry or "w" not in entry:
            raise SkinError(f"{name} entries must be dicts with 'bone' and 'w'")
        bone = entry["bone"]
        weight = entry["w"]
        if not isinstance(bone, int) or isinstance(bone, bool):
            raise SkinError(f"{name} bone index must be an integer")
        if not _is_number(weight):
            raise SkinError(f"{name} weight must be a number")
        parsed.append({"bone": bone, "w": float(weight)})
    return parsed


def _check_bone_indices(weights: list[dict], bone_count: int) -> None:
    for item in weights:
        bone = item["bone"]
        if bone < 0 or bone >= bone_count:
            raise SkinError(
                f"bone index {bone} out of range [0, {bone_count})"
            )


def _matrix_3x3(matrix: list[float]) -> list[float]:
    """取行主序 4x4 矩阵左上 3x3，仍按行主序返回 9 个元素。"""
    return [
        matrix[0], matrix[1], matrix[2],
        matrix[4], matrix[5], matrix[6],
        matrix[8], matrix[9], matrix[10],
    ]


def _det_3x3(m: list[float]) -> float:
    return (
        m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
    )


def _inverse_transpose_3x3(m: list[float]) -> list[float]:
    """返回 3x3 行主序矩阵的逆转置（9 个元素）；奇异时抛 SkinError。"""
    a, b, c = m[0], m[1], m[2]
    d, e, f = m[3], m[4], m[5]
    g, h, i = m[6], m[7], m[8]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) <= _DET_EPS:
        raise SkinError("matrix is singular and cannot be inverted")

    inv_det = 1.0 / det
    # inv = adj / det，invT = adj^T / det。
    return [
        (e * i - f * h) * inv_det, (f * g - d * i) * inv_det, (d * h - e * g) * inv_det,
        (c * h - b * i) * inv_det, (a * i - c * g) * inv_det, (b * g - a * h) * inv_det,
        (b * f - c * e) * inv_det, (c * d - a * f) * inv_det, (a * e - b * d) * inv_det,
    ]


def _mat3_apply(matrix: list[float], vector: list[float]) -> list[float]:
    x, y, z = vector
    return [
        matrix[0] * x + matrix[1] * y + matrix[2] * z,
        matrix[3] * x + matrix[4] * y + matrix[5] * z,
        matrix[6] * x + matrix[7] * y + matrix[8] * z,
    ]


def _normalize_vector(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 0.0 or not math.isfinite(length):
        return [0.0, 0.0, 0.0]
    return [component / length for component in vector]


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def normalize_weights(weights: list[dict]) -> list[dict]:
    """归一化骨骼权重，并最多保留 4 个影响。

    参数:
        weights: ``[{"bone": int, "w": number}, ...]``。不得原地修改。

    契约:
        - 丢弃 ``w <= 0`` 的条目；按 ``w`` 从大到小保留最多 4 个影响。
        - 保留项的 ``w`` 重新缩放到总和为 1，允许误差 1e-6。
        - 若没有正权重，返回空 list，表示该顶点退化为静止姿态。
        - 返回新 list；每项保留输入的 ``bone``，``w`` 为归一化后的 float。
    """
    parsed = _require_weights(weights)

    positive = [item for item in parsed if item["w"] > 0.0]
    if not positive:
        return []

    positive.sort(key=lambda item: item["w"], reverse=True)
    kept = positive[:_MAX_INFLUENCES]
    total = sum(item["w"] for item in kept)
    if total <= 0.0 or not math.isfinite(total):
        return []

    return [{"bone": item["bone"], "w": item["w"] / total} for item in kept]


def apply_bone_matrix(pos: list[float], matrix: list[float]) -> list[float]:
    """用行主序 4x4 矩阵变换一个三维点。

    参数:
        pos: 长度 3 的坐标，按齐次坐标 ``(x, y, z, 1)`` 处理。
        matrix: 长度 16 的扁平 list，行主序；平移在 ``m[3]``、``m[7]``、
            ``m[11]``。

    契约:
        按 ``x' = m[0]*x + m[1]*y + m[2]*z + m[3]`` 等计算，返回
        ``[x', y', z']``；矩阵需为仿射矩阵，不做透视除法。数值容差 1e-6。
    """
    point = _require_vector(pos, 3, "pos")
    m = _require_matrix(matrix)

    x, y, z = point
    return [
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    ]


def skin_vertex(vertex: dict, matrices: list[list[float]]) -> dict:
    """对单个顶点做线性混合蒙皮。

    参数:
        vertex: 顶点 dict，必须包含 ``pos``、``normal`` 和 ``weights``。
        matrices: 每根骨骼一个行主序 4x4 扁平 list；传入前必须已经完成
            ``pose_matrix @ inverse_bind_matrix`` 的预乘。

    契约:
        - 权重按 :func:`normalize_weights` 处理：总和为 1（误差 1e-6），
          最多保留 4 个影响。
        - 位置为各骨骼 ``M_i * pos`` 的加权和；法线用各 ``M_i`` 左上 3x3
          的逆转置变换后再加权，不能直接使用原矩阵乘法线。
        - 没有正权重时退化为静止姿态，返回输入 ``pos``/``normal`` 的副本。
        - 缺少字段、形状非法或骨骼索引越界时抛 :class:`SkinError`。
        - 返回新 dict ``{"pos": [...], "normal": [...]}``，结果容差 1e-6。
    """
    if not isinstance(vertex, dict):
        raise SkinError("vertex must be a dict")
    for key in ("pos", "normal", "weights"):
        if key not in vertex:
            raise SkinError(f"vertex is missing '{key}'")

    pos = _require_vector(vertex["pos"], 3, "pos")
    normal = _require_vector(vertex["normal"], 3, "normal")
    raw_weights = _require_weights(vertex["weights"])

    if not isinstance(matrices, (list, tuple)):
        raise SkinError("matrices must be a list")
    _check_bone_indices(raw_weights, len(matrices))

    weights = normalize_weights(raw_weights)
    if not weights:
        # 退化到静止姿态：返回副本，避免把顶点塌到原点。
        return {"pos": list(pos), "normal": list(normal)}

    # 每个顶点独立累加，避免跨顶点共享缓冲。
    out_pos = [0.0, 0.0, 0.0]
    for item in weights:
        bone = item["bone"]
        weight = item["w"]
        transformed = apply_bone_matrix(pos, matrices[bone])
        for axis in range(3):
            out_pos[axis] += weight * transformed[axis]

    out_normal = skin_normal(normal, matrices, weights)
    return {"pos": out_pos, "normal": out_normal}


def skin_normal(
    normal: list[float],
    matrices: list[list[float]],
    weights: list[dict],
) -> list[float]:
    """使用逆转置矩阵变换并混合法线。

    参数:
        normal: 输入法线，无需预先归一化。
        matrices: 每根骨骼一个行主序 4x4 扁平 list。
        weights: 与 :func:`normalize_weights` 相同的权重结构。

    契约:
        - 权重归一化到 1（误差 1e-6），最多保留 4 个影响。
        - 对每个影响使用矩阵左上 3x3 的逆转置变换 ``normal``（忽略平移），
          再按权重求和并归一化；不能直接用原矩阵乘法线。
        - 没有正权重时返回归一化后的输入法线。
        - 矩阵奇异时抛 :class:`SkinError`。
        - 返回长度为 3 的 list，数值容差 1e-6。
    """
    source = _require_vector(normal, 3, "normal")
    if not isinstance(matrices, (list, tuple)):
        raise SkinError("matrices must be a list")
    raw_weights = _require_weights(weights)
    _check_bone_indices(raw_weights, len(matrices))

    normalized = normalize_weights(raw_weights)
    if not normalized:
        return _normalize_vector(source)

    out_normal = [0.0, 0.0, 0.0]
    for item in normalized:
        bone = item["bone"]
        weight = item["w"]
        m = _require_matrix(matrices[bone])
        inverse_transpose = _inverse_transpose_3x3(_matrix_3x3(m))
        transformed = _mat3_apply(inverse_transpose, source)
        for axis in range(3):
            out_normal[axis] += weight * transformed[axis]

    return _normalize_vector(out_normal)


def inverse_bind(bind_matrix: list[float]) -> list[float]:
    """求 4x4 绑定矩阵的逆。

    参数:
        bind_matrix: 行主序、长度 16 的扁平 list。

    契约:
        - 返回长度 16 的行主序逆矩阵，计算误差不超过 1e-6。
        - 若矩阵奇异（行列式在数值上为 0，例如 ``abs(det) <= 1e-12``）或
          长度不为 16，抛 :class:`SkinError`。
        - 不得修改输入。
    """
    m = _require_matrix(bind_matrix, "bind_matrix")
    a, b, c, d = m[0], m[1], m[2], m[3]
    e, f, g, h = m[4], m[5], m[6], m[7]
    i, j, k, l = m[8], m[9], m[10], m[11]
    n, o, p, q = m[12], m[13], m[14], m[15]

    # 展开代数余子式矩阵与行列式（行主序）。
    cofactor = [
        f * (k * q - l * p) - g * (j * q - l * o) + h * (j * p - k * o),
        -(e * (k * q - l * p) - g * (i * q - l * n) + h * (i * p - k * n)),
        e * (j * q - l * o) - f * (i * q - l * n) + h * (i * o - j * n),
        -(e * (j * p - k * o) - f * (i * p - k * n) + g * (i * o - j * n)),
        -(b * (k * q - l * p) - c * (j * q - l * o) + d * (j * p - k * o)),
        a * (k * q - l * p) - c * (i * q - l * n) + d * (i * p - k * n),
        -(a * (j * q - l * o) - b * (i * q - l * n) + d * (i * o - j * n)),
        a * (j * p - k * o) - b * (i * p - k * n) + c * (i * o - j * n),
        b * (g * q - h * p) - c * (f * q - h * o) + d * (f * p - g * o),
        -(a * (g * q - h * p) - c * (e * q - h * n) + d * (e * p - g * n)),
        a * (f * q - h * o) - b * (e * q - h * n) + d * (e * o - f * n),
        -(a * (f * p - g * o) - b * (e * p - g * n) + c * (e * o - f * n)),
        -(b * (g * l - h * k) - c * (f * l - h * j) + d * (f * k - g * j)),
        a * (g * l - h * k) - c * (e * l - h * i) + d * (e * k - g * i),
        -(a * (f * l - h * j) - b * (e * l - h * i) + d * (e * j - f * i)),
        a * (f * k - g * j) - b * (e * k - g * i) + c * (e * j - f * i),
    ]

    # 行列式等于第一行与对应代数余子式的点积。
    det = a * cofactor[0] + b * cofactor[1] + c * cofactor[2] + d * cofactor[3]
    if abs(det) <= _DET_EPS:
        raise SkinError("bind matrix is singular and cannot be inverted")

    inv_det = 1.0 / det
    # adjugate 是代数余子式矩阵的转置；上面对 cofactor 已按行存代数余子式，
    # 因此这里转置写入逆矩阵。
    inverse = [cofactor[col * 4 + row] * inv_det for row in range(4) for col in range(4)]
    return inverse


def blend_shapes(
    positions: list[list[float]],
    shapes: list[list[list[float]]],
    weights: list[float],
) -> list[list[float]]:
    """叠加形变目标（blend shapes）。

    参数:
        positions: 基础顶点位置，形如 ``[[x, y, z], ...]``。
        shapes: 形变目标位置列表；每个 target 与 ``positions`` 等长。
        weights: 每个 target 对应一个标量权重。

    契约:
        - 返回 ``base + sum(clamp(w_i, 0, 1) * (target_i - base))`` 的新
          位置列表。
        - 权重和可以不为 1，也不重新归一化；负权重钳到 0，超过 1 的钳到 1，
          以避免翻面或过冲。
        - 结果容差 1e-6；长度不一致时抛 :class:`SkinError`。
    """
    if not isinstance(positions, (list, tuple)):
        raise SkinError("positions must be a list")
    if not isinstance(shapes, (list, tuple)):
        raise SkinError("shapes must be a list")
    if not isinstance(weights, (list, tuple)):
        raise SkinError("weights must be a list")

    base = [_require_vector(point, 3, "position") for point in positions]

    if len(shapes) != len(weights):
        raise SkinError("shapes and weights must have the same length")

    result = [list(point) for point in base]
    for shape, weight in zip(shapes, weights):
        if not _is_number(weight):
            raise SkinError("shape weight must be a number")
        if not isinstance(shape, (list, tuple)) or len(shape) != len(base):
            raise SkinError("each shape must have the same length as positions")

        clamped = min(1.0, max(0.0, float(weight)))
        if clamped == 0.0:
            continue

        target = [_require_vector(point, 3, "shape point") for point in shape]
        for index, point in enumerate(target):
            for axis in range(3):
                result[index][axis] += clamped * (point[axis] - base[index][axis])

    return result


def validate_skin(vertex: dict, bone_count: int) -> list[str]:
    """检查顶点蒙皮数据并返回问题描述列表。

    参数:
        vertex: 顶点 dict，至少应包含 ``pos``、``normal`` 和 ``weights``。
        bone_count: 可用骨骼矩阵数量。

    契约:
        - 原始权重和不是 1（容差 1e-6）时，加入
          ``"weights must sum to 1"``。
        - 正权重影响数超过 4 时，加入
          ``"too many influences (max 4)"``。
        - 每个 ``bone`` 不在 ``[0, bone_count)`` 时，加入包含
          ``"out of range"`` 的消息。
        - 返回按上述顺序排列的 ``list[str]``；合法时返回空 list。
        - 本函数只报告问题，不抛 :class:`SkinError`，也不修改输入。
    """
    issues: list[str] = []

    weights = []
    if isinstance(vertex, dict):
        raw = vertex.get("weights", [])
        if isinstance(raw, (list, tuple)):
            for entry in raw:
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("bone"), int)
                    and not isinstance(entry.get("bone"), bool)
                    and _is_number(entry.get("w"))
                ):
                    weights.append({"bone": entry["bone"], "w": float(entry["w"])})

    total = sum(item["w"] for item in weights)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_SUM_EPS):
        issues.append("weights must sum to 1")

    positive_count = sum(1 for item in weights if item["w"] > 0.0)
    if positive_count > _MAX_INFLUENCES:
        issues.append("too many influences (max 4)")

    for item in weights:
        bone = item["bone"]
        if bone < 0 or bone >= bone_count:
            issues.append(
                f"bone index {bone} out of range [0, {bone_count})"
            )

    return issues


def skin_batch(
    vertices: list[dict],
    matrices: list[list[float]],
    config: dict,
) -> dict:
    """批量蒙皮，失败时不产生部分结果。

    参数:
        vertices: 顶点 dict 的 list。
        matrices: 骨骼皮肤矩阵列表（行主序、扁平 4x4）。
        config: 配置 dict；``config.get("validate", True)`` 为 True 时先用
            :func:`validate_skin` 做整体预检，默认严格模式。

    契约:
        - 默认先校验所有顶点；任一顶点有问题立即抛 :class:`SkinError`，
          绝不返回部分结果，也不修改输入。
        - 每个顶点使用独立的累加缓冲，禁止跨顶点共享位置/法线累加器。
        - 成功时返回 ``{"vertices": [...], "count": n}``，顺序与输入一致；
          每个元素是 :func:`skin_vertex` 的结果。
        - 每个顶点的权重和为 1（误差 1e-6）、最多 4 个影响；空输入返回
          ``{"vertices": [], "count": 0}``。
    """
    if not isinstance(vertices, (list, tuple)):
        raise SkinError("vertices must be a list")
    if not isinstance(matrices, (list, tuple)):
        raise SkinError("matrices must be a list")

    config = config or {}
    strict = bool(config.get("validate", True)) if isinstance(config, dict) else True

    if strict:
        for index, vertex in enumerate(vertices):
            issues = validate_skin(vertex, len(matrices))
            if issues:
                raise SkinError(
                    f"vertex {index} is invalid: " + "; ".join(issues)
                )

    # 全部通过预检后才开始计算；每个顶点在 skin_vertex 内部使用独立缓冲。
    results = [skin_vertex(vertex, matrices) for vertex in vertices]
    return {"vertices": results, "count": len(results)}
