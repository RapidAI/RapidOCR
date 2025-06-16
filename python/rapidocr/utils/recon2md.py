import numpy as np
from .output import RapidOCROutput


def get_box_properties(box: np.ndarray) -> dict:
    """从坐标数组中计算框的几何属性"""
    # box shape is (4, 2) -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    ys = box[:, 1]
    xs = box[:, 0]

    top = np.min(ys)
    bottom = np.max(ys)
    left = np.min(xs)

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "height": bottom - top,
        "center_y": top + (bottom - top) / 2,
    }


def reconstruct_layout_to_markdown(result: RapidOCROutput) -> str:
    """
    根据 OCR 结果的坐标信息，将文本还原为近似原始排版的 Markdown。

    Args:
        result (RapidOCROutput): RapidOCR 的输出结果对象。

    Returns:
        str: 模拟原始排版的 Markdown 字符串。
    """
    if not result or not result.txts:
        return "没有检测到任何文本。"

    # 1. 将 box 和 text 绑定并排序
    #    主键：box 的顶部 y 坐标；次键：box 的左侧 x 坐标
    combined_data = sorted(
        zip(result.boxes, result.txts),
        key=lambda item: (
            get_box_properties(item[0])["top"],
            get_box_properties(item[0])["left"],
        ),
    )

    output_lines = []
    if not combined_data:
        return ""

    # 初始化当前行和前一个框的属性
    current_line_parts = [combined_data[0][1]]
    prev_props = get_box_properties(combined_data[0][0])

    # 从第二个框开始遍历
    for box, text in combined_data[1:]:
        current_props = get_box_properties(box)

        # 启发式规则来决定如何布局
        # 条件1：中心线距离是否足够近
        min_height = min(current_props["height"], prev_props["height"])
        centers_are_close = abs(current_props["center_y"] - prev_props["center_y"]) < (
            min_height * 0.5
        )

        # 条件2：是否存在垂直方向的重叠
        # 计算重叠区域的顶部和底部
        overlap_top = max(prev_props["top"], current_props["top"])
        overlap_bottom = min(prev_props["bottom"], current_props["bottom"])
        has_vertical_overlap = overlap_bottom > overlap_top

        # 最终判断：满足任一条件即可
        is_same_line = centers_are_close or has_vertical_overlap

        if is_same_line:
            # 在同一行，用空格隔开
            current_line_parts.append("   ")  # 使用多个空格以产生明显间距
            current_line_parts.append(text)
        else:
            # 不在同一行，需要换行
            # 先将上一行组合成字符串并添加到输出列表
            output_lines.append("".join(current_line_parts))

            # 规则2：判断是否需要插入空行（新段落）
            # 如果垂直间距大于上一个框高度的某个比例（如70%），则认为是一个新段落
            vertical_gap = current_props["top"] - prev_props["bottom"]
            if vertical_gap > prev_props["height"] * 0.7:
                output_lines.append("")  # 插入空行来创建段落

            # 开始一个新行
            current_line_parts = [text]

        # 更新前一个框的属性
        prev_props = current_props

    # 添加最后一行
    output_lines.append("".join(current_line_parts))

    return "\n".join(output_lines)


if __name__ == "__main__":
    """
    使用方法:
    from rapidocr.utils.recon2md import reconstruct_layout_to_markdown
    engine = RapidOCR()
    img_url = "no_git_oic/test.png"

    result = engine(img_url)
    markdown_layout = reconstruct_layout_to_markdown(result)
    print(markdown_layout)
    """
    mock_result = RapidOCROutput(
        img=np.array([]),
        # 提供了原始的 boxes 数据用于排版分析
        boxes=np.array(
            [
                [[6.0, 2.0], [322.0, 9.0], [320.0, 104.0], [4.0, 97.0]],  # 正品促销
                [
                    [70.0, 98.0],
                    [252.0, 98.0],
                    [252.0, 125.0],
                    [70.0, 125.0],
                ],  # 大桶装更划算
                [
                    [68.0, 144.0],
                    [256.0, 144.0],
                    [256.0, 165.0],
                    [68.0, 165.0],
                ],  # 强力去污符合国标
                [
                    [108.0, 170.0],
                    [217.0, 170.0],
                    [217.0, 182.0],
                    [108.0, 182.0],
                ],  # 40°C深度防冻不结冰 (这里数据不全，仅作示意)
                # 手动添加一些模拟的 box-text 对
                # 模拟 "日常价￥ 10.0起" 在同一行
                [
                    [10.0, 200.0],
                    [80.0, 200.0],
                    [80.0, 220.0],
                    [10.0, 220.0],
                ],  # 日常价￥
                [
                    [90.0, 200.0],
                    [160.0, 200.0],
                    [160.0, 220.0],
                    [90.0, 220.0],
                ],  # 10.0起
                # 模拟 "底价 5.8" 在下一行
                [[10.0, 230.0], [60.0, 230.0], [60.0, 250.0], [10.0, 250.0]],  # 底价
                [[70.0, 230.0], [120.0, 230.0], [120.0, 250.0], [70.0, 250.0]],  # 5.8
                # 模拟一个大间距之后的文本
                [
                    [10.0, 300.0],
                    [150.0, 300.0],
                    [150.0, 320.0],
                    [10.0, 320.0],
                ],  # 惊喜福利不容错过
            ],
            dtype=np.float32,
        ),
        txts=(
            "正品促销",
            "大桶装更划算",
            "强力去污符合国标",
            "40°C深度防冻不结冰",
            "日常价￥",
            "10.0起",
            "底价",
            "5.8",
            "惊喜福利不容错过",
        ),
        scores=(0.99, 0.98, 0.97, 0.93, 0.81, 0.99, 0.99, 0.99, 0.99),
    )

    # 2. 调用函数进行转换
    markdown_layout = reconstruct_layout_to_markdown(mock_result)

    # 3. 打印还原排版后的 Markdown 字符串
    print(markdown_layout)
