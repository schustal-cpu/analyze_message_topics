# src/visualization.py

import pandas as pd

def base_table_style(df, caption):
    styles = [
        {
            "selector": "table",
            "props": [("width", "100%")]
        },
        {
            "selector": "th",
            "props": [("text-align", "center"), ("font-weight", "bold")]
        },
        {
            "selector": "td",
            "props": [("padding", "4px"), ("border-right", "1px solid lightgray")]
        },
        {
            "selector": "caption",
            "props": [
                ("caption-side", "top"),
                ("text-align", "center"),
                ("font-weight", "bold"),   # ← wichtig
                ("font-size", "16px")
            ]
        }
    ]

    styled = (
        df.style
        .hide(axis="index")
        .set_table_styles(styles)
    )

    if caption:
        styled = styled.set_caption(caption)

    return styled


def style_topic_sentiment(
    df,
    caption,
    pos_threshold=0.3,
    neg_threshold=-0.3,
    pos_color="#c6efce",
    neg_color="#ffc7ce"
):
    def highlight_sentiment(val):
        if val >= pos_threshold:
            return f"background-color: {pos_color}"
        elif val <= neg_threshold:
            return f"background-color: {neg_color}"
        return ""

    styled = base_table_style(df, caption)

    styled = styled.map(
        highlight_sentiment,
        subset=["avg_sentiment"]
    )

    return styled


def style_topic_comparison(df, caption):
    return base_table_style(df, caption)


def style_compare_top_tokens(df, caption):
    styled = base_table_style(df, caption)

    styled = styled.background_gradient(
        cmap="Blues",
        subset=[col for col in df.columns if "Count" in str(col)]
    )

    return styled