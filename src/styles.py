# src/visualization.py

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


def style_compare_top_tokens(df, caption):
    styled = base_table_style(df, caption)

    styled = styled.background_gradient(
        cmap="Blues",
        subset=[col for col in df.columns if "Count" in str(col)]
    )

    return styled

def style_tuning_eval_generic(
    df,
    style_metrics,
    score_col="overall_score",
    hide_cols=None
):
    display_df = df.drop(columns=hide_cols or [], errors="ignore")

    styler = display_df.style

    format_dict = {
        score_col: "{:.4f}"
    }

    for metric in style_metrics:
        col = metric["col"]
        higher_is_better = metric["higher_is_better"]
        good_quantile = metric["good_quantile"]
        bad_quantile = metric["bad_quantile"]
        fmt = metric.get("format", "{:.4f}")

        good_value = display_df[col].quantile(good_quantile)
        bad_value = display_df[col].quantile(bad_quantile)

        def color_value(v, good_value=good_value, bad_value=bad_value, higher_is_better=higher_is_better):
            if higher_is_better:
                if v >= good_value:
                    return "background-color: #c6efce"
                elif v <= bad_value:
                    return "background-color: #ffc7ce"
            else:
                if v <= good_value:
                    return "background-color: #c6efce"
                elif v >= bad_value:
                    return "background-color: #ffc7ce"

            return "background-color: #ffeb9c"

        styler = styler.map(color_value, subset=[col])
        format_dict[col] = fmt

    return styler.format(format_dict)