"""
Airline Delay Analytics - Interactive Dashboard (Plotly Dash)
Student : A.V.P.S. Akshaya | Roll No: CB.SC.U4CSE24401 | Course: 23CSE353
Dataset : Flight Delay and Causes (Kaggle, undersc0re)

Run locally :  python app.py   ->  http://127.0.0.1:8050
Deploy      :  Plotly Cloud / any host that runs a Dash app (exposes `server`)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

# ----------------------------------------------------------------------------
# 1. LOAD + CLEAN DATA  (same steps as the notebook)
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent
CANDIDATES = ["Flight_delay.csv", "Flight_delay(1).csv", "flight_delay.csv"]
DATA_PATH = next((HERE / c for c in CANDIDATES if (HERE / c).exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Put the dataset CSV (Flight_delay.csv) in the same folder as app.py."
    )

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
for col in ["Org_Airport", "Dest_Airport"]:
    df[col] = df[col].fillna("Unknown airport")

CAUSE_COLS = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
CAUSE_LABELS = {
    "CarrierDelay": "Carrier",
    "WeatherDelay": "Weather",
    "NASDelay": "NAS",
    "SecurityDelay": "Security",
    "LateAircraftDelay": "Late aircraft",
}
NUM_COLS = [
    "DayOfWeek", "ArrDelay", "DepDelay", "Distance", "ActualElapsedTime",
    "CRSElapsedTime", "AirTime", "TaxiIn", "TaxiOut",
] + CAUSE_COLS
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[CAUSE_COLS] = df[CAUSE_COLS].fillna(0)
df = df.dropna(subset=["Date", "ArrDelay", "DepDelay"])
df["DayName"] = df["Date"].dt.day_name()

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CORR_COLS = [
    "ArrDelay", "DepDelay", "Distance", "ActualElapsedTime", "CRSElapsedTime",
    "AirTime", "TaxiIn", "TaxiOut",
] + CAUSE_COLS
CORR_COLS = [c for c in CORR_COLS if c in df.columns]

AIRLINES = sorted(df["Airline"].unique())
DATE_MIN, DATE_MAX = df["Date"].min().date(), df["Date"].max().date()
IS_DELAYED_SUBSET = bool((df["ArrDelay"] >= 15).all())

# ----------------------------------------------------------------------------
# 2. STYLE
# ----------------------------------------------------------------------------
NAVY, BLUE, SKY, ORANGE, RED, GREEN = "#0B2545", "#13599E", "#4EA8DE", "#F4A259", "#D1495B", "#2A9D8F"
BG, CARD, TEXT, MUTED = "#F2F5FA", "#FFFFFF", "#1B2A41", "#6B7A90"
PALETTE = [BLUE, SKY, ORANGE, GREEN, RED, "#8E7DBE", "#F6C453", "#5C677D"]
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

CARD_STYLE = {
    "background": CARD, "borderRadius": "12px", "padding": "14px 16px",
    "boxShadow": "0 1px 4px rgba(11,37,69,.10)",
}


def style_fig(fig, height=340, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=45, r=15, t=50, b=40),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT, color=TEXT, size=12),
        title=dict(font=dict(size=15, color=NAVY), x=0.01),
        showlegend=legend, hoverlabel=dict(font_size=12),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor="#E7ECF3", zeroline=False)
    fig.update_yaxes(gridcolor="#E7ECF3", zeroline=False)
    return fig


def empty_fig(msg="No data for the selected filters"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig)


def card(children, **extra):
    return html.Div(children, style={**CARD_STYLE, **extra})


def graph(gid, **kw):
    return dcc.Graph(id=gid, config={"displaylogo": False}, **kw)


def section(title, subtitle):
    return html.Div(
        [html.H2(title, style={"margin": "0", "color": NAVY, "fontSize": "20px"}),
         html.P(subtitle, style={"margin": "2px 0 0", "color": MUTED, "fontSize": "13px"})],
        style={"margin": "26px 0 10px"},
    )


def kpi(label, value_id, color):
    return html.Div(
        [html.Div(label, style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "letterSpacing": ".5px"}),
         html.Div("-", id=value_id, style={"fontSize": "26px", "fontWeight": 700, "color": color, "marginTop": "4px"})],
        style={**CARD_STYLE, "borderLeft": f"5px solid {color}"},
    )


def grid(cols, children, gap="14px"):
    return html.Div(children, style={"display": "grid", "gridTemplateColumns": cols, "gap": gap})


# ----------------------------------------------------------------------------
# 3. STATIC FIGURES  (misleading vs ethical - they ignore filters on purpose)
# ----------------------------------------------------------------------------
_air = (df.groupby("Airline", as_index=False)["ArrDelay"].mean()
          .sort_values("ArrDelay", ascending=False).head(6).sort_values("ArrDelay"))


def _bar(truncate):
    fig = go.Figure(go.Bar(
        x=_air["Airline"], y=_air["ArrDelay"],
        marker_color=RED if truncate else GREEN,
        text=_air["ArrDelay"].round(1), textposition="outside" if not truncate else "none",
        hovertemplate="%{x}<br>%{y:.2f} min<extra></extra>",
    ))
    lo, hi = _air["ArrDelay"].min(), _air["ArrDelay"].max()
    fig.update_yaxes(range=[lo - 2, hi + 2] if truncate else [0, hi * 1.15],
                     title="Mean arrival delay (min)")
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(title="MISLEADING: truncated y-axis" if truncate
                      else "ETHICAL: y-axis starts at zero")
    return style_fig(fig, height=380)


FIG_MISLEADING, FIG_ETHICAL = _bar(True), _bar(False)

# ----------------------------------------------------------------------------
# 4. LAYOUT
# ----------------------------------------------------------------------------
app = Dash(__name__, title="Airline Delay Analytics")
server = app.server  # for deployment

app.layout = html.Div(
    style={"background": BG, "minHeight": "100vh", "fontFamily": FONT, "color": TEXT},
    children=[
        # ---- header
        html.Div(
            style={"background": f"linear-gradient(120deg,{NAVY},{BLUE})", "color": "white",
                   "padding": "22px 32px"},
            children=[
                html.H1("✈ Airline Delay Analytics", style={"margin": 0, "fontSize": "28px"}),
                html.Div("Flight delay patterns, airline & airport comparison, and recorded causes",
                         style={"opacity": .85, "marginTop": "4px"}),
                html.Div("A.V.P.S. Akshaya · CB.SC.U4CSE24401 · 23CSE353 Data Visualization",
                         style={"opacity": .65, "fontSize": "12px", "marginTop": "6px"}),
            ],
        ),
        html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 20px 40px"}, children=[

            # ---- problem statement
            card([
                html.H2("Problem statement", style={"margin": "0 0 6px", "color": NAVY, "fontSize": "18px"}),
                html.P(
                    "Flight delays impact airline efficiency and passenger experience. This project "
                    "aims to analyze flight delay trends, compare airline and airport performance, "
                    "identify major delay causes, and visualize relationships between flight variables "
                    "using an interactive dashboard.",
                    style={"margin": 0, "fontSize": "14px", "lineHeight": "1.6"},
                ),
            ], marginTop="16px", borderLeft=f"5px solid {SKY}"),

            # ---- filters (sticky)
            html.Div(
                style={**CARD_STYLE, "position": "sticky", "top": "8px", "zIndex": 50,
                       "marginTop": "16px"},
                children=grid("1.2fr 2fr 1.5fr 1.3fr", [
                    html.Div([html.Label("Date range", style={"fontSize": "12px", "color": MUTED}),
                              dcc.DatePickerRange(id="f-date", min_date_allowed=DATE_MIN,
                                                  max_date_allowed=DATE_MAX, start_date=DATE_MIN,
                                                  end_date=DATE_MAX, display_format="DD MMM YYYY")]),
                    html.Div([html.Label("Airlines (empty = all)", style={"fontSize": "12px", "color": MUTED}),
                              dcc.Dropdown(id="f-airline", options=AIRLINES, multi=True,
                                           placeholder="All airlines")]),
                    html.Div([html.Label("Day of week", style={"fontSize": "12px", "color": MUTED}),
                              dcc.Dropdown(id="f-day", options=DAY_ORDER, multi=True,
                                           placeholder="All days")]),
                    html.Div([html.Label("Trend granularity", style={"fontSize": "12px", "color": MUTED}),
                              dcc.RadioItems(id="f-gran", options=["Daily", "Weekly", "Monthly"],
                                             value="Daily", inline=True,
                                             inputStyle={"marginLeft": "10px", "marginRight": "4px"})]),
                ]),
            ),

            # ---- data coverage banner
            (html.Div(
                "⚠ Data note: every record in this file has an arrival delay ≥ 15 minutes, so it is a "
                "delayed-flight subset. Averages describe delayed flights only, not all flights.",
                style={"background": "#FFF4E0", "border": f"1px solid {ORANGE}", "borderRadius": "10px",
                       "padding": "10px 14px", "marginTop": "14px", "fontSize": "13px"})
             if IS_DELAYED_SUBSET else html.Div()),

            # ---- KPIs
            html.Div(style={"marginTop": "14px"}, children=grid("repeat(6,1fr)", [
                kpi("Flight records", "k-records", NAVY),
                kpi("Mean arrival delay", "k-mean", BLUE),
                kpi("Median arrival delay", "k-median", SKY),
                kpi("90th percentile", "k-p90", ORANGE),
                kpi("Dep-Arr correlation", "k-corr", GREEN),
                kpi("Leading cause", "k-cause", RED),
            ])),

            # ---- insights
            html.Div(id="insights", style={**CARD_STYLE, "marginTop": "14px", "fontSize": "14px",
                                           "lineHeight": "1.6", "borderLeft": f"5px solid {ORANGE}"}),

            # ---- 1. Time
            section("1 · Delay trend over time", "How do average and median arrival delays change across the period?"),
            grid("2fr 1fr", [card(graph("g-trend")), card(graph("g-dow"))]),

            # ---- 2. Airlines & airports
            section("2 · Airlines & airports", "Which carriers and origin airports experience the longest delays?"),
            grid("1fr 1fr", [card(graph("g-airline")), card(graph("g-airport"))]),
            html.Div(style={"marginTop": "14px"}, children=card(graph("g-box"))),

            # ---- 3. Causes
            section("3 · Causes of delay", "Which of the five recorded causes account for the most delay minutes?"),
            grid("1.3fr 1fr", [card(graph("g-cause")), card(graph("g-cause-air"))]),

            # ---- 4. Distribution & relationships
            section("4 · Distribution & relationships",
                    "How skewed are delays, and how strongly does departure delay carry into arrival delay?"),
            grid("1fr 1fr", [card(graph("g-hist")), card(graph("g-scatter"))]),
            html.Div(style={"marginTop": "14px"}, children=card(graph("g-heat"))),

            # ---- 5. Ethics
            section("5 · Ethical visualization: misleading vs honest",
                    "Same data, two charts. These two ignore the filters on purpose."),
            grid("1fr 1fr", [card(dcc.Graph(figure=FIG_MISLEADING, config={"displaylogo": False})),
                             card(dcc.Graph(figure=FIG_ETHICAL, config={"displaylogo": False}))]),
            card([
                html.B("How the left chart misleads: "),
                "the y-axis starts just below the smallest bar, so a gap of a few minutes looks like "
                "the top airline is several times worse than the others. Bar length no longer represents "
                "the value. ",
                html.B("Fix: "),
                "start bar charts at zero, label values, and show sample sizes so differences are judged "
                "in proportion.",
            ], marginTop="14px", fontSize="14px", lineHeight="1.6"),

            # ---- limitations
            section("Limitations", ""),
            card(html.Ul([
                html.Li("Subset only: arrival delay ≥ 15 min in every row, so on-time/early flights are absent."),
                html.Li("Cancelled/Diverted fields carry no variation in this extract, so no cancellation analysis."),
                html.Li("Cause columns are recorded minutes, not proof of causation. Correlation ≠ causation."),
                html.Li("Airlines and airports differ in routes and sample size; raw means are not a fair ranking."),
                html.Li("Scatter and box plots use random samples for speed; KPIs use all filtered records."),
            ], style={"margin": 0, "paddingLeft": "18px", "fontSize": "14px", "lineHeight": "1.7"})),

            html.Div("Source: Kaggle - Flight Delay and Causes (undersc0re). Built with Plotly Dash.",
                     style={"textAlign": "center", "color": MUTED, "fontSize": "12px", "marginTop": "26px"}),
        ]),
    ],
)

# ----------------------------------------------------------------------------
# 5. CALLBACK
# ----------------------------------------------------------------------------
GRAPH_IDS = ["g-trend", "g-dow", "g-airline", "g-airport", "g-box", "g-cause",
             "g-cause-air", "g-hist", "g-scatter", "g-heat"]
KPI_IDS = ["k-records", "k-mean", "k-median", "k-p90", "k-corr", "k-cause"]


@app.callback(
    [Output(i, "figure") for i in GRAPH_IDS]
    + [Output(i, "children") for i in KPI_IDS]
    + [Output("insights", "children")],
    Input("f-date", "start_date"), Input("f-date", "end_date"),
    Input("f-airline", "value"), Input("f-day", "value"), Input("f-gran", "value"),
)
def update(start, end, airlines, days, gran):
    d = df[(df["Date"] >= start) & (df["Date"] <= end)]
    if airlines:
        d = d[d["Airline"].isin(airlines)]
    if days:
        d = d[d["DayName"].isin(days)]

    if d.empty:
        return ([empty_fig()] * len(GRAPH_IDS)) + ["-"] * len(KPI_IDS) + ["No records match the filters."]

    # ---- KPIs
    corr = d["DepDelay"].corr(d["ArrDelay"])
    cause_tot = d[CAUSE_COLS].sum().sort_values(ascending=False)
    top_cause = CAUSE_LABELS[cause_tot.index[0]]
    kpis = [f"{len(d):,}", f"{d['ArrDelay'].mean():.1f} min", f"{d['ArrDelay'].median():.1f} min",
            f"{d['ArrDelay'].quantile(.9):.1f} min", f"{corr:.2f}", top_cause]

    # ---- 1. trend
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
    tr = (d.set_index("Date")["ArrDelay"].resample(freq).agg(["mean", "median", "size"])
            .dropna().reset_index())
    f_trend = go.Figure()
    f_trend.add_trace(go.Scatter(x=tr["Date"], y=tr["mean"], name="Mean", mode="lines+markers" if gran != "Daily" else "lines",
                                 line=dict(color=BLUE, width=2), customdata=tr["size"],
                                 hovertemplate="%{x|%d %b %Y}<br>Mean %{y:.1f} min<br>%{customdata:,} records<extra></extra>"))
    f_trend.add_trace(go.Scatter(x=tr["Date"], y=tr["median"], name="Median", mode="lines",
                                 line=dict(color=ORANGE, width=1.6, dash="dot"),
                                 hovertemplate="%{x|%d %b %Y}<br>Median %{y:.1f} min<extra></extra>"))
    f_trend.update_layout(title=f"{gran} arrival delay trend", yaxis_title="Delay (minutes)")
    f_trend.update_xaxes(rangeslider_visible=True)
    style_fig(f_trend, height=400, legend=True)

    # ---- day of week
    dow = (d.groupby("DayName")["ArrDelay"].mean().reindex(DAY_ORDER).dropna().reset_index())
    f_dow = px.bar(dow, x="DayName", y="ArrDelay", title="Mean arrival delay by weekday",
                   labels={"DayName": "", "ArrDelay": "Minutes"})
    f_dow.update_traces(marker_color=SKY, hovertemplate="%{x}<br>%{y:.1f} min<extra></extra>")
    f_dow.update_yaxes(rangemode="tozero")
    f_dow.update_xaxes(tickangle=-30)
    style_fig(f_dow, height=400)

    # ---- 2. airlines
    a = (d.groupby("Airline").agg(Mean=("ArrDelay", "mean"), Records=("ArrDelay", "size"))
           .reset_index().sort_values("Mean"))
    f_air = go.Figure(go.Bar(x=a["Mean"], y=a["Airline"], orientation="h", marker_color=BLUE,
                             customdata=a["Records"],
                             hovertemplate="%{y}<br>Mean %{x:.1f} min<br>n = %{customdata:,}<extra></extra>"))
    f_air.update_layout(title="Mean arrival delay by airline", xaxis_title="Minutes")
    f_air.update_xaxes(rangemode="tozero")
    style_fig(f_air, height=max(340, 26 * len(a) + 90))
    f_air.update_layout(margin=dict(l=170))

    # ---- airports (>=500 records, or relax for small filtered slices)
    thr = 500 if len(d) > 100_000 else max(30, int(len(d) * 0.002))
    ap = (d.groupby(["Origin", "Org_Airport"]).agg(Mean=("ArrDelay", "mean"), Records=("ArrDelay", "size"))
            .reset_index())
    ap = ap[ap["Records"] >= thr].sort_values("Mean", ascending=False).head(12).sort_values("Mean")
    if ap.empty:
        f_ap = empty_fig("Not enough records per airport")
    else:
        f_ap = go.Figure(go.Bar(
            x=ap["Mean"], y=ap["Origin"] + " · " + ap["Org_Airport"].str.slice(0, 28), orientation="h",
            marker_color=ORANGE, customdata=ap["Records"],
            hovertemplate="%{y}<br>Mean %{x:.1f} min<br>n = %{customdata:,}<extra></extra>"))
        f_ap.update_layout(title=f"Top origin airports by mean delay (≥ {thr} records)", xaxis_title="Minutes")
        f_ap.update_xaxes(rangemode="tozero")
        style_fig(f_ap, height=max(340, 26 * len(ap) + 90))
        f_ap.update_layout(margin=dict(l=240))

    # ---- boxplot (top 8 airlines in selection; sampled)
    top8 = d["Airline"].value_counts().head(8).index
    bx = d[d["Airline"].isin(top8)]
    bx = bx.sample(n=min(30_000, len(bx)), random_state=42)
    f_box = px.box(bx, x="Airline", y="ArrDelay", color="Airline", points=False,
                   title="Arrival delay spread across the most frequent airlines (outliers hidden)",
                   labels={"ArrDelay": "Arrival delay (min)"}, color_discrete_sequence=PALETTE)
    f_box.update_xaxes(tickangle=-25)
    style_fig(f_box, height=420)

    # ---- 3. causes
    c = pd.DataFrame({"Cause": [CAUSE_LABELS[i] for i in cause_tot.index],
                      "Minutes": cause_tot.values})
    c["Share"] = 100 * c["Minutes"] / c["Minutes"].sum()
    f_cause = go.Figure(go.Bar(x=c["Cause"], y=c["Minutes"], marker_color=PALETTE[:len(c)],
                               text=c["Share"].round(1).astype(str) + "%", textposition="outside",
                               hovertemplate="%{x}<br>%{y:,.0f} min<extra></extra>"))
    f_cause.update_layout(title="Total recorded delay minutes by cause", yaxis_title="Total minutes")
    f_cause.update_yaxes(rangemode="tozero")
    style_fig(f_cause, height=380)

    # cause mix per airline (100% stacked, top 8 airlines)
    mix = d[d["Airline"].isin(top8)].groupby("Airline")[CAUSE_COLS].sum()
    mix = mix.div(mix.sum(axis=1), axis=0) * 100
    f_mix = go.Figure()
    for i, col in enumerate(CAUSE_COLS):
        f_mix.add_trace(go.Bar(y=mix.index, x=mix[col], name=CAUSE_LABELS[col], orientation="h",
                               marker_color=PALETTE[i],
                               hovertemplate="%{y}<br>" + CAUSE_LABELS[col] + ": %{x:.1f}%<extra></extra>"))
    f_mix.update_layout(barmode="stack", title="Cause mix per airline (% of cause minutes)",
                        xaxis_title="% of airline's recorded cause minutes")
    style_fig(f_mix, height=380, legend=True)
    f_mix.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(l=160, b=80))

    # ---- 4. histogram
    cap = d["ArrDelay"].quantile(.99)
    f_hist = px.histogram(d[d["ArrDelay"] <= cap], x="ArrDelay", nbins=60,
                          title="Distribution of arrival delays (up to 99th percentile)",
                          labels={"ArrDelay": "Arrival delay (min)"})
    f_hist.update_traces(marker_color=BLUE, marker_line_color="white", marker_line_width=.5)
    f_hist.add_vline(x=d["ArrDelay"].median(), line_dash="dash", line_color=ORANGE,
                     annotation_text="median", annotation_position="top")
    f_hist.update_layout(yaxis_title="Flight records")
    style_fig(f_hist, height=380)

    # scatter (sampled)
    sc = d.sample(n=min(8000, len(d)), random_state=42)
    f_sc = px.scatter(sc, x="DepDelay", y="ArrDelay", color="Airline", opacity=.55,
                      hover_data=["Date", "Origin", "Dest", "Distance"],
                      title=f"Departure vs arrival delay (sample of {len(sc):,}, r = {corr:.2f})",
                      labels={"DepDelay": "Departure delay (min)", "ArrDelay": "Arrival delay (min)"},
                      color_discrete_sequence=PALETTE)
    f_sc.update_traces(marker=dict(size=5))
    style_fig(f_sc, height=380, legend=True)
    f_sc.update_layout(legend=dict(font=dict(size=9)))

    # heatmap
    cm = d[CORR_COLS].corr().round(2)
    f_heat = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1,
                                  colorscale="RdBu_r", text=cm.values, texttemplate="%{text}",
                                  textfont=dict(size=10), hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    f_heat.update_layout(title="Correlation heatmap of operational and delay variables")
    f_heat.update_yaxes(autorange="reversed")
    f_heat.update_xaxes(tickangle=-35)
    style_fig(f_heat, height=560)

    # ---- insights text
    a_sorted = a.sort_values("Mean", ascending=False)
    worst, best = a_sorted.iloc[0], a_sorted.iloc[-1]
    ac = abs(corr)
    strength = "strong" if ac > .7 else "moderate" if ac > .4 else "weak"
    direction = "positive" if corr > 0 else "negative"
    mean_v, med_v = d["ArrDelay"].mean(), d["ArrDelay"].median()
    skew_note = (f"Mean ({mean_v:.1f}) is above median ({med_v:.1f}) minutes, so a few very long delays pull the average up."
                 if mean_v > med_v * 1.1 else f"Mean ({mean_v:.1f}) and median ({med_v:.1f}) minutes are close.")
    insights = [
        html.B("Key insights for the current selection: "),
        f"{worst['Airline']} has the highest mean delay ({worst['Mean']:.1f} min, n={worst['Records']:,}) and "
        f"{best['Airline']} the lowest ({best['Mean']:.1f} min, n={best['Records']:,}). "
        f"{top_cause} delay is the largest recorded cause ({c['Share'].iloc[0]:.1f}% of cause minutes). "
        f"Departure and arrival delays have a {strength} {direction} correlation (r = {corr:.2f}). "
        + skew_note,
    ]

    figs = [f_trend, f_dow, f_air, f_ap, f_box, f_cause, f_mix, f_hist, f_sc, f_heat]
    return figs + kpis + [insights]


if __name__ == "__main__":
    app.run(debug=True)
