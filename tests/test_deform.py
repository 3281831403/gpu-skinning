"""蒙皮契约测试：10 个缺陷回归 + 5 个组合/边界回归。"""

import math

import pytest

from skin.deform import (
    SkinError,
    apply_bone_matrix,
    blend_shapes,
    inverse_bind,
    normalize_weights,
    skin_batch,
    skin_normal,
    skin_vertex,
    validate_skin,
)

EPS = 1e-6


def assert_vec_close(actual, expected, tol=EPS):
    """逐分量比较两个向量。"""
    assert len(actual) == len(expected), (actual, expected)
    for got, want in zip(actual, expected):
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=tol), (actual, expected)


def identity():
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def translation(x, y, z):
    return [
        1.0, 0.0, 0.0, float(x),
        0.0, 1.0, 0.0, float(y),
        0.0, 0.0, 1.0, float(z),
        0.0, 0.0, 0.0, 1.0,
    ]


def scale(x, y, z):
    return [
        float(x), 0.0, 0.0, 0.0,
        0.0, float(y), 0.0, 0.0,
        0.0, 0.0, float(z), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def rotation_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def test_normalize_weights_scales_to_one():
    """缺陷 1：权重不归一导致模型整体缩放偏大。"""
    raw = [
        {"bone": 0, "w": 1.0},
        {"bone": 1, "w": 0.5},
        {"bone": 2, "w": 0.5},
    ]

    got = normalize_weights(raw)
    by_bone = {item["bone"]: item["w"] for item in got}

    assert len(got) == 3
    assert math.isclose(sum(item["w"] for item in got), 1.0, abs_tol=EPS)
    assert math.isclose(by_bone[0], 0.5, abs_tol=EPS)
    assert math.isclose(by_bone[1], 0.25, abs_tol=EPS)
    assert math.isclose(by_bone[2], 0.25, abs_tol=EPS)
    assert raw[0]["w"] == 1.0, "normalize_weights 不应修改输入"


def test_normalize_weights_keeps_at_most_four_influences():
    """缺陷 2：保留超过 4 个影响，突破硬件上限。"""
    raw = [
        {"bone": 0, "w": 0.1},
        {"bone": 1, "w": 0.2},
        {"bone": 2, "w": 0.3},
        {"bone": 3, "w": 0.4},
        {"bone": 4, "w": 0.5},
    ]

    got = normalize_weights(raw)

    assert len(got) == 4
    assert {item["bone"] for item in got} == {1, 2, 3, 4}
    assert [item["w"] for item in got] == pytest.approx(
        [0.5 / 1.4, 0.4 / 1.4, 0.3 / 1.4, 0.2 / 1.4],
        abs=EPS,
    )
    assert math.isclose(sum(item["w"] for item in got), 1.0, abs_tol=EPS)


def test_skin_normal_uses_inverse_transpose():
    """缺陷 3：法线直接乘矩阵，非均匀缩放下方向错误。"""
    scale_x = scale(2.0, 1.0, 1.0)
    got = skin_normal(
        [1.0, 1.0, 0.0],
        [scale_x],
        [{"bone": 0, "w": 1.0}],
    )

    inv_len = math.sqrt(1.25)
    assert_vec_close(got, [0.5 / inv_len, 1.0 / inv_len, 0.0])


def test_inverse_bind_is_applied_before_bone_pose():
    """缺陷 4：逆绑定矩阵未预乘，模型整体偏移。"""
    bind = translation(2.0, 0.0, 0.0)
    pose = rotation_z(math.pi / 2.0)
    bind_space_pos = [3.0, 0.0, 0.0]

    local = apply_bone_matrix(bind_space_pos, inverse_bind(bind))
    got = apply_bone_matrix(local, pose)

    assert_vec_close(got, [0.0, 1.0, 0.0])


def test_inverse_bind_rejects_singular_matrix():
    """缺陷 5：奇异矩阵不求逆，继续计算产生 NaN。"""
    singular = scale(1.0, 0.0, 1.0)

    with pytest.raises(SkinError):
        inverse_bind(singular)


def test_blend_shapes_clamps_weights():
    """缺陷 6：形变权重不钳制，负权重导致模型翻面。"""
    base = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    shapes = [
        [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0]],
        [[0.0, 4.0, 0.0], [0.0, 5.0, 0.0]],
    ]

    got = blend_shapes(base, shapes, [-1.0, 1.5])

    assert_vec_close(got[0], [0.0, 4.0, 0.0])
    assert_vec_close(got[1], [0.0, 5.0, 0.0])


def test_validate_skin_reports_out_of_range_bone():
    """缺陷 7：骨骼索引越界不报错，随后读到错误矩阵。"""
    vertex = {
        "pos": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "weights": [
            {"bone": 0, "w": 0.5},
            {"bone": 3, "w": 0.5},
        ],
    }

    issues = validate_skin(vertex, 2)

    assert any("out of range" in issue for issue in issues), issues


def test_skin_batch_isolates_vertices_and_rolls_back():
    """缺陷 8：批量共用累加缓冲，顶点之间串数据；同时验证失败回滚。"""
    bone = translation(1.0, 0.0, 0.0)
    vertices = [
        {
            "pos": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "weights": [{"bone": 0, "w": 1.0}],
        },
        {
            "pos": [5.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "weights": [{"bone": 0, "w": 1.0}],
        },
    ]
    original = [
        {
            "pos": list(vertex["pos"]),
            "normal": list(vertex["normal"]),
            "weights": [dict(weight) for weight in vertex["weights"]],
        }
        for vertex in vertices
    ]

    result = skin_batch(vertices, [bone], {})

    assert result["count"] == 2
    assert_vec_close(result["vertices"][0]["pos"], [1.0, 0.0, 0.0])
    assert_vec_close(result["vertices"][1]["pos"], [6.0, 0.0, 0.0])

    bad = {
        "pos": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "weights": [{"bone": 5, "w": 1.0}],
    }
    with pytest.raises(SkinError):
        skin_batch(vertices + [bad], [bone], {})
    assert vertices == original, "批量失败后不得修改输入"


def test_skin_vertex_zero_weights_returns_rest_pose():
    """缺陷 9：权重全零未退化为静止姿态，位置变成 0。"""
    vertex = {
        "pos": [3.0, 4.0, 5.0],
        "normal": [0.0, 1.0, 0.0],
        "weights": [
            {"bone": 0, "w": 0.0},
            {"bone": 1, "w": 0.0},
        ],
    }

    got = skin_vertex(vertex, [scale(2.0, 2.0, 2.0), translation(10.0, 0.0, 0.0)])

    assert_vec_close(got["pos"], vertex["pos"])
    assert_vec_close(got["normal"], vertex["normal"])


def test_apply_bone_matrix_uses_row_major_layout():
    """缺陷 10：行主序/列主序混用，平移落到错误分量。"""
    matrix = [
        1.0, 0.0, 0.0, 10.0,
        0.0, 1.0, 0.0, 20.0,
        0.0, 0.0, 1.0, 30.0,
        0.0, 0.0, 0.0, 1.0,
    ]

    got = apply_bone_matrix([1.0, 2.0, 3.0], matrix)

    assert_vec_close(got, [11.0, 22.0, 33.0])


def test_single_bone_weight_one_matches_matrix():
    """组合：单骨权重 1 时，位置等于矩阵变换结果，法线保持一致。"""
    matrix = [
        0.0, -1.0, 0.0, 5.0,
        1.0, 0.0, 0.0, -1.0,
        0.0, 0.0, 1.0, 2.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    vertex = {
        "pos": [1.0, 2.0, 3.0],
        "normal": [0.0, 1.0, 0.0],
        "weights": [{"bone": 0, "w": 1.0}],
    }

    got = skin_vertex(vertex, [matrix])

    assert_vec_close(got["pos"], apply_bone_matrix(vertex["pos"], matrix))
    assert_vec_close(got["pos"], [3.0, 0.0, 5.0])
    assert_vec_close(got["normal"], skin_normal(vertex["normal"], [matrix], vertex["weights"]))


def test_identity_skinning_keeps_vertex():
    """组合：单位矩阵蒙皮后顶点位置和法线不变。"""
    vertex = {
        "pos": [1.0, 2.0, 3.0],
        "normal": [0.0, 0.6, 0.8],
        "weights": [{"bone": 0, "w": 1.0}],
    }

    got = skin_vertex(vertex, [identity()])

    assert_vec_close(got["pos"], vertex["pos"])
    assert_vec_close(got["normal"], vertex["normal"])


def test_skin_vertex_rejects_empty_vertex():
    """组合：空顶点必须抛 SkinError，而不是 IndexError/KeyError。"""
    with pytest.raises(SkinError):
        skin_vertex({}, [identity()])


def test_normalized_weights_sum_to_one_within_tolerance():
    """组合：归一化后的权重和在 1e-6 容差内等于 1。"""
    got = normalize_weights(
        [
            {"bone": 0, "w": 3.0},
            {"bone": 1, "w": 1.0},
        ]
    )

    total = sum(item["w"] for item in got)
    assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(got[0]["w"], 0.75, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(got[1]["w"], 0.25, rel_tol=0.0, abs_tol=1e-6)


def test_skin_batch_preserves_order():
    """组合：批量全部成功时，输出顺序与输入一致。"""
    vertices = [
        {
            "pos": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "weights": [{"bone": 0, "w": 1.0}],
        },
        {
            "pos": [1.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "weights": [{"bone": 0, "w": 1.0}],
        },
        {
            "pos": [2.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "weights": [{"bone": 0, "w": 1.0}],
        },
    ]

    result = skin_batch(vertices, [translation(10.0, 0.0, 0.0)], {})

    assert result["count"] == 3
    assert len(result["vertices"]) == 3
    for index, item in enumerate(result["vertices"]):
        assert_vec_close(item["pos"], [10.0 + index, 0.0, 0.0])
