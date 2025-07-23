import os
from pymongo import MongoClient
import pandas as pd

from dash import Dash, html, dcc, Output, Input
import plotly.express as px
from datetime import datetime, timedelta, timezone
from datetime import timedelta

# ── Load config ────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb+srv://dbMechConnect:mWw7W5R70NN3tLU@cluster0.3trcy.mongodb.net/mech-connect-lott-dev"
DB_NAME = os.getenv("DB_NAME", "mech-connect-lott-dev")

# ── MongoDB client ─────────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
suggestions_col = client[DB_NAME]['suggestions']
chatthreads_col = client[DB_NAME]['chatthreads']

# ── Data fetchers ──────────────────────────────────────────────────────────────

def fetch_top_issues(limit: int = 5) -> pd.DataFrame:
    pipeline = [
        {"$unwind": "$problemsReasons"},
        {"$group": {
            "_id": "$problemsReasons.title",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    results = list(suggestions_col.aggregate(pipeline))
    if not results:
        return pd.DataFrame(columns=["issue", "count"])
    return pd.DataFrame(results).rename(columns={"_id": "issue"})


def fetch_total_chats() -> int:
    return chatthreads_col.count_documents({})


def fetch_avg_steps_last_week(days: int = 7) -> pd.DataFrame:
    """
    For each of the last `days` days, compute the average number of messages
    in the chatthreads collection.
    """
    cutoff = datetime.utcnow() - timedelta(days=days-1)
    pipeline = [
        {"$match": {"created_at": {"$gte": cutoff}}},
        {"$project": {
            "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "count": {"$size": {"$ifNull": ["$messages", []]}}
        }},
        {"$group": {"_id": "$day", "avg_steps": {"$avg": "$count"}}},
        {"$sort": {"_id": 1}}
    ]
    res = list(chatthreads_col.aggregate(pipeline))
    if not res:
        return pd.DataFrame(columns=["day", "avg_steps"])
    df = pd.DataFrame(res).rename(columns={"_id": "day"})
    df["day"] = pd.to_datetime(df["day"])
    return df

def fetch_top_makes(limit: int = 5) -> pd.DataFrame:
    pipeline = [
        # 1) Only include threads with a non-empty manufacturer
        {"$match": {
            "vehicle_info.manufacturer": {
                "$exists": True,
                "$nin": [None, ""]
            }
        }},
        # 2) Group & count by that manufacturer
        {"$group": {
            "_id": "$vehicle_info.manufacturer",
            "count": {"$sum": 1}
        }},
        # 3) Sort & limit
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    results = list(chatthreads_col.aggregate(pipeline))
    if not results:
        return pd.DataFrame(columns=["make", "count"])
    df = pd.DataFrame(results).rename(columns={"_id": "make"})
    return df


# ── Dash app setup ─────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "MechConnect Dashboard"

app.layout = html.Div(
    style={"backgroundColor": "#F5F5F5", "minHeight": "100vh"},
    children=[
        # Navbar
        html.Nav(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "10px 20px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=html.Img(src="/assets/mechconnect_logo.png",
                              style={"height": "40px"})
        ),

        # Main content
        html.Div(
            style={"padding": "20px"},
            children=[
                # KPI at top
                html.Div(
                    style={
                        "backgroundColor": "#EC6936",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "color": "#FFFFFF",
                        "marginBottom": "40px",
                        "width": "100%"
                    },
                    children=[
                        html.H4("Total Chats Initiated", style={"margin": 0}),
                        html.H1(id="total-chats", children="0",
                                style={"margin": 0, "fontSize": "48px"})
                    ]
                ),
                
                  # ── Top Diagnosed Vehicles ─────────────────────────────────
                html.H3("Top Diagnosed Vehicles", style={"marginBottom": "20px"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        # Left: Top 5 Car Makes Diagnosed
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px"
                            },
                            children=[
                                html.H4("Top Car Makes Diagnosed", style={"marginTop": 0}),
                                dcc.Graph(id="top-makes-chart", style={"height": "360px", "width": "100%"})
                            ]
                        ),
                        # Right: placeholder
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "color": "#888",
                                "fontStyle": "italic"
                            },
                            children=[html.Div("Coming Soon")]
                        )
                    ]
                ),

                # Common Issues and Parts
                html.H3("Common Issues and Parts",
                        style={"marginBottom": "20px"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px"
                            },
                            children=[
                                html.H4("Top 5 Diagnosed Issues",
                                        style={"marginTop": 0}),
                                dcc.Graph(
                                    id="top-issues-chart",
                                    style={"height": "360px", "width": "100%"}
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "color": "#888",
                                "fontStyle": "italic"
                            },
                            children=[html.Div("Coming Soon")]
                        ),
                    ]
                ),

                # Diagnostic Behavior
                html.H3("Diagnostic Behavior", style={
                        "marginTop": "40px", "marginBottom": "20px"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        # Avg Steps line chart
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px"
                            },
                            children=[
                                html.H4("Avg. Steps/Questions per Diagnosis",
                                        style={"marginTop": 0}),
                                dcc.Graph(id="avg-steps-chart",
                                          style={"height": "360px"})
                            ]
                        ),
                        # placeholders
                        *[
                            html.Div(
                                style={
                                    "flex": "1",
                                    "backgroundColor": "#FFFFFF",
                                    "padding": "20px",
                                    "borderRadius": "8px",
                                    "textAlign": "center",
                                    "color": "#888",
                                    "fontStyle": "italic"
                                },
                                children=[html.Div("Coming Soon")]
                            )
                            for _ in range(2)
                        ]
                    ]
                ),

                # Auto-refresh
                dcc.Interval(id="interval-component",
                             interval=2*60*1000, n_intervals=0)
            ]
        )
    ]
)

# ── Callback to refresh everything every 5 minutes ─────────────────────────────


# ── Callback to refresh all charts/KPIs ─────────────────────────────────────────
@app.callback(
    [
      Output("top-makes-chart", "figure"),
      Output("top-issues-chart","figure"),
      Output("total-chats",      "children"),
      Output("avg-steps-chart",  "figure"),
    ],
    Input("interval-component", "n_intervals")
)
def update_dashboard(n):
    # 1) Top 5 makes
    df_makes = fetch_top_makes()
    if df_makes.empty:
        fig_makes = px.bar(title="No data")
    else:
        fig_makes = px.bar(
            df_makes, x="count", y="make", orientation="h",
            labels={"count": "Occurrences", "make": "Make"},
            height=360
        )
        fig_makes.update_traces(marker_color="#E8E8E8", marker_line_width=0)
        fig_makes.update_layout(
            yaxis={"autorange": "reversed"}, bargap=0.2,
            margin={"l": 140, "r": 20, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        # right-edge border
        y0, y1 = fig_makes.layout.yaxis.domain
        span   = (y1 - y0) / max(len(df_makes),1)
        for i, row in df_makes.iterrows():
            cnt    = row["count"]
            center = y1 - span*(i+0.5)
            half   = span*0.4
            fig_makes.add_shape(
                type="line", x0=cnt, x1=cnt,
                y0=center-half, y1=center+half,
                xref="x", yref="paper",
                line=dict(color="#EC6936", width=3)
            )

    # 2) Top 5 issues (reuse previous logic)
    df_issues = fetch_top_issues()
    if df_issues.empty:
        fig_issues = px.bar(title="No diagnosed issues yet")
        fig_issues.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        fig_issues = px.bar(
            df_issues, x="count", y="issue", orientation="h",
            labels={"count": "Occurrences", "issue": "Issue"},
            height=360
        )
        fig_issues.update_traces(marker_color="#E8E8E8", marker_line_width=0)
        fig_issues.update_layout(
            yaxis={"autorange": "reversed"}, bargap=0.2,
            margin={"l": 140, "r": 20, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        y0, y1 = fig_issues.layout.yaxis.domain
        span   = (y1 - y0) / max(len(df_issues),1)
        for i, row in df_issues.iterrows():
            cnt    = row["count"]
            center = y1 - span*(i+0.5)
            half   = span*0.4
            fig_issues.add_shape(
                type="line", x0=cnt, x1=cnt,
                y0=center-half, y1=center+half,
                xref="x", yref="paper",
                line=dict(color="#EC6936", width=3)
            )

    # 3) Total chats
    total_chats = fetch_total_chats()

    # 4) Avg steps chart
    df_avg = fetch_avg_steps_last_week()
    if df_avg.empty:
        fig_avg = px.line(title="No data")
        fig_avg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        fig_avg = px.line(
            df_avg, x="day", y="avg_steps",
            labels={"day": "", "avg_steps": "Avg Steps"},
            height=360
        )
        fig_avg.update_traces(line=dict(color="#EC6936"), marker=dict(size=4))
        fig_avg.update_layout(
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        fig_avg.update_xaxes(tickformat="%a", dtick="D1")

    return fig_makes, fig_issues, total_chats, fig_avg

# ── Run server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
