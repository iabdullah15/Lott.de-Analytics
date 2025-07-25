import os
from pymongo import MongoClient
import pandas as pd
import json
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
from datetime import datetime, timedelta, timezone
from datetime import timedelta

from google.analytics.data import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
from google.oauth2 import service_account
from dotenv import load_dotenv

# path to your service-account JSON
KEY_PATH = "ga-credentials.json"
PROPERTY_ID = "484780893"  # e.g. "123456789"

# credentials = service_account.Credentials.from_service_account_file(
#     KEY_PATH,
#     scopes=["https://www.googleapis.com/auth/analytics.readonly"]
# )
# ga_client = BetaAnalyticsDataClient(credentials=credentials)


# Load variables from .env
load_dotenv()

# Now read the path to your JSON key
KEY_PATH = os.environ.get("GA_CREDENTIALS_JSON")
if not KEY_PATH:
    raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS env var")

from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient

credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/analytics.readonly"]
)
ga_client = BetaAnalyticsDataClient(credentials=credentials)


# ── Load config ────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb+srv://dbMechConnect:mWw7W5R70NN3tLU@cluster0.3trcy.mongodb.net/mech-connect-lott-dev"
DB_NAME = os.getenv("DB_NAME", "mech-connect-prod")

# ── MongoDB client ─────────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
suggestions_col = client[DB_NAME]['suggestions']
chatthreads_col = client[DB_NAME]['chatthreads']

# ── Data fetchers ──────────────────────────────────────────────────────────────


def fetch_top_issues(window: str = "all", limit: int = 5) -> pd.DataFrame:
    """
    window: one of "7d", "30d", "90d", or "all"
    """
    pipeline = []
    # 1) optionally filter by created_at
    if window != "all":
        days = int(window[:-1])
        cutoff = datetime.utcnow() - timedelta(days=days)
        pipeline.append({"$match": {"created_at": {"$gte": cutoff}}})

    # 2) unwind, group, sort, limit
    pipeline += [
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


# def fetch_avg_steps_last_week(days: int = 7) -> pd.DataFrame:
#     """
#     For each of the last `days` days, compute the average number of messages
#     in the chatthreads collection.
#     """
#     cutoff = datetime.utcnow() - timedelta(days=days-1)
#     pipeline = [
#         {"$match": {"created_at": {"$gte": cutoff}}},
#         {"$project": {
#             "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
#             "count": {"$size": {"$ifNull": ["$messages", []]}}
#         }},
#         {"$group": {"_id": "$day", "avg_steps": {"$avg": "$count"}}},
#         {"$sort": {"_id": 1}}
#     ]
#     res = list(chatthreads_col.aggregate(pipeline))
#     if not res:
#         return pd.DataFrame(columns=["day", "avg_steps"])
#     df = pd.DataFrame(res).rename(columns={"_id": "day"})
#     df["day"] = pd.to_datetime(df["day"])
#     return df


def fetch_avg_steps(window: str = "7d") -> pd.DataFrame:
    """
    Compute average # messages per diagnosis over a rolling window:
      - "7d"  → last 7 days
      - "30d" → last 30 days
      - "90d" → last 90 days
      - "all" → no time filter
    """
    # build optional match stage
    pipeline = []
    if window != "all":
        days = int(window[:-1])
        cutoff = datetime.utcnow() - timedelta(days=days-1)
        pipeline.append({"$match": {"created_at": {"$gte": cutoff}}})

    # project day & message count, then group/avg/sort
    pipeline += [
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


def fetch_top_models(limit: int = 5) -> pd.DataFrame:
    pipeline = [
        # only docs where series exists and is non-empty
        {"$match": {
            "vehicle_info.series": {"$exists": True, "$nin": [None, ""]}
        }},
        # group & count
        {"$group": {
            "_id": "$vehicle_info.series",
            "count": {"$sum": 1}
        }},
        # sort & limit
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    res = list(chatthreads_col.aggregate(pipeline))
    if not res:
        return pd.DataFrame(columns=["model", "count"])
    return pd.DataFrame(res).rename(columns={"_id": "model"})


# def fetch_completion_rate() -> pd.DataFrame:
#     """
#     Returns a DataFrame with two rows:
#     - status: "Completed" or "Incomplete"
#     - count: number of sessions
#     """
#     total = chatthreads_col.count_documents({})
#     completed = suggestions_col.count_documents({
#         "problemsReasons": {"$exists": True, "$ne": []},
#         "carParts":       {"$exists": True, "$ne": []},
#         "effectedParts":  {"$exists": True}
#     })
#     incomplete = max(total - completed, 0)
#     return pd.DataFrame([
#         {"status": "Completed",   "count": completed},
#         {"status": "Incomplete",  "count": incomplete}
#     ])

def fetch_completion_rate(window: str = "all") -> pd.DataFrame:
    """
    Returns a DataFrame with two rows:
     - status: "Completed" or "Incomplete"
     - count: number of sessions
    Filters both chatthreads & suggestions by created_at >= cutoff if window != "all".
    """
    # build optional match filter
    match = {}
    if window != "all":
        days = int(window[:-1])
        cutoff = datetime.utcnow() - timedelta(days=days-1)
        match = {"created_at": {"$gte": cutoff}}

    # total sessions in window
    total = chatthreads_col.count_documents(match)

    # completed sessions in window
    completed_match = {
        **match,
        "problemsReasons": {"$exists": True, "$ne": []},
        "carParts":       {"$exists": True, "$ne": []},
        "effectedParts":  {"$exists": True}
    }
    completed = suggestions_col.count_documents(completed_match)

    incomplete = max(total - completed, 0)
    return pd.DataFrame([
        {"status": "Completed",   "count": completed},
        {"status": "Incomplete",  "count": incomplete}
    ])


def fetch_avg_diagnosis_time_by_month() -> pd.DataFrame:
    """
    Join suggestions -> chatthreads, compute (suggestion.created_at - thread.created_at)
    in hours, then average per month.
    """
    pipeline = [
        # Only suggestions that reference a chatThread
        {"$match": {"chatThread": {"$exists": True}}},
        # Lookup the chatthreads document
        {"$lookup": {
            "from": "chatthreads",
            "localField": "chatThread",
            "foreignField": "_id",
            "as": "thread"
        }},
        {"$unwind": "$thread"},
        # Compute month and diff in hours
        {"$project": {
            "month": {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}},
            "durationHours": {
                "$divide": [
                    {"$subtract": ["$created_at", "$thread.created_at"]},
                    1000 * 60
                ]
            }
        }},
        {"$group": {
            "_id": "$month",
            "avg_time": {"$avg": "$durationHours"}
        }},
        {"$sort": {"_id": 1}}
    ]
    res = list(suggestions_col.aggregate(pipeline))
    if not res:
        return pd.DataFrame(columns=["month", "avg_time"])
    df = pd.DataFrame(res).rename(columns={"_id": "month"})
    df["month"] = pd.to_datetime(df["month"] + "-01")
    return df


def fetch_top_parts(window: str = "all", limit: int = 5) -> pd.DataFrame:
    """
    window: one of "7d", "30d", "90d", or "all"
    """
    pipeline = []

    # 1) optionally filter by created_at
    if window != "all":
        days = int(window[:-1])
        cutoff = datetime.utcnow() - timedelta(days=days)
        pipeline.append({"$match": {"created_at": {"$gte": cutoff}}})

    # 2) unwind, group, sort, limit
    pipeline += [
        {"$unwind": "$carParts"},
        {"$group": {
            "_id": "$carParts.title",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]

    res = list(suggestions_col.aggregate(pipeline))
    if not res:
        return pd.DataFrame(columns=["part", "count"])
    return pd.DataFrame(res).rename(columns={"_id": "part"})


def fetch_hourly_active_users():
    now = datetime.utcnow().date()
    yesterday = now - timedelta(days=1)
    request = RunReportRequest(
        property=f"properties/{PROPERTY_ID}",
        dimensions=[Dimension(name="hour")],
        metrics=[Metric(name="activeUsers")],
        date_ranges=[DateRange(start_date=str(yesterday), end_date=str(now))]
    )
    resp = ga_client.run_report(request)
    # parse into pandas
    data = [(row.dimension_values[0].value, int(row.metric_values[0].value))
            for row in resp.rows]
    df = pd.DataFrame(data, columns=["hour", "users"])
    df["hour"] = df["hour"].astype(int)  # 0–23
    return df


def fetch_active_by_weekday(days: int = 28) -> pd.DataFrame:
    """
    1) Query GA for activeUsers by date (last `days` days).
    2) In pandas, map dates → weekday abbreviations.
    3) Group & average (or sum) per weekday in calendar order.
    """
    today = datetime.utcnow().date()
    start = today - timedelta(days=days)

    request = RunReportRequest(
        property=f"properties/{PROPERTY_ID}",
        dimensions=[Dimension(name="date")],
        metrics=[Metric(name="activeUsers")],
        date_ranges=[DateRange(start_date=str(start), end_date=str(today))]
    )
    resp = ga_client.run_report(request)

    # build a DataFrame of daily counts
    rows = [
        (r.dimension_values[0].value, int(r.metric_values[0].value))
        for r in resp.rows
    ]
    df = pd.DataFrame(rows, columns=["date", "users"])
    if df.empty:
        return pd.DataFrame(columns=["day", "users"])

    # convert and extract weekday abbreviations
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day_name().str[:3]  # Mon, Tue, ...

    # group (you can sum or average; here we average)
    df2 = (
        df.groupby("day", sort=False)["users"]
        .sum()
        .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        .reset_index()
    )
    return df2


# ── Dash app setup ─────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "MechConnect Dashboard"

server = app.server

app.layout = html.Div(
    style={"backgroundColor": "#F5F5F5", "minHeight": "100vh",
           "fontFamily": "'Inter', sans-serif"},
    children=[
        # Navbar
        html.Nav(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "10px 20px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=html.Img(
                src="/assets/mechconnect_logo.png",
                style={"height": "40px"}
            )
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
                        html.H1(
                            id="total-chats",
                            children="0",
                            style={"margin": 0, "fontSize": "48px"}
                        )
                    ]
                ),

                # Top Diagnosed Vehicles
                html.H3("Top Diagnosed Vehicles",
                        style={"marginBottom": "20px"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        # Top 5 Car Makes Diagnosed
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px"
                            },
                            children=[
                                html.H4("Top Car Makes Diagnosed",
                                        style={"marginTop": 0}),
                                dcc.Graph(
                                    id="top-makes-chart",
                                    style={"height": "360px", "width": "100%"}
                                )
                            ]
                        ),
                        # Top 5 Car Models Diagnosed
                        html.Div(
                            style={"flex": "1",
                                   "backgroundColor": "#FFFFFF",
                                   "padding": "20px",
                                   "borderRadius": "8px"
                                   },
                            children=[
                                html.H4("Top 5 Car Models Diagnosed",
                                        style={"marginTop": 0}),
                                dcc.Graph(id="top-models-chart",
                                          style={"height": "360px", "width": "100%"})
                            ]
                        )
                    ]
                ),

                # ── Common Issues and Parts ────────────────────────────────────────────────
                html.H3("Common Issues and Parts",
                        style={"marginTop": "40px", "marginBottom": "20px"}),

                # Wrap both cards in a single flex container
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        # Card 1: Top 5 Diagnosed Issues with dropdown
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px"
                            },
                            children=[
                                # header row
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "center",
                                        "marginBottom": "10px"
                                    },
                                    children=[
                                        html.H4("Top 5 Diagnosed Issues",
                                                style={"margin": 0}),
                                        dcc.Dropdown(
                                            id="issues-window-dropdown",
                                            options=[
                                                {"label": "Last 7 days",
                                                    "value": "7d"},
                                                {"label": "Last 30 days",
                                                    "value": "30d"},
                                                {"label": "Last 90 days",
                                                    "value": "90d"},
                                                {"label": "All time",
                                                    "value": "all"},
                                            ],
                                            value="all",
                                            clearable=False,
                                            style={"width": "150px"}
                                        )
                                    ]
                                ),
                                # the bar chart
                                dcc.Graph(
                                    id="top-issues-chart",
                                    style={"height": "360px", "width": "100%"}
                                ),
                            ]
                        ),

                        # Card 2: Top Selling Parts (Most Clicked)
                        html.Div(
                            style={"flex": "1", "backgroundColor": "#FFFFFF",
                                   "padding": "20px", "borderRadius": "8px"},
                            children=[
                                # header row: title + dropdown
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "center",
                                        "marginBottom": "10px"
                                    },
                                    children=[
                                        html.H4("Top Selling Parts (Most Clicked)",
                                                style={"margin": 0}),
                                        dcc.Dropdown(
                                            id="parts-window-dropdown",
                                            options=[
                                                {"label": "Last 7 days",
                                                    "value": "7d"},
                                                {"label": "Last 30 days",
                                                    "value": "30d"},
                                                {"label": "Last 90 days",
                                                    "value": "90d"},
                                                {"label": "All time",
                                                    "value": "all"},
                                            ],
                                            value="all",
                                            clearable=False,
                                            style={"width": "150px"}
                                        )
                                    ]
                                ),

                                # the bar chart
                                dcc.Graph(id="top-parts-chart",
                                          style={"height": "360px", "width": "100%"})
                            ]
                        ),
                    ]
                ),

                # Diagnostic Behavior
                html.H3("Diagnostic Behavior", style={
                        "marginTop": "40px", "marginBottom": "20px"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        # Avg Steps per diagnosis line chart
                        html.Div(
                            style={
                                "flex": "1",
                                "backgroundColor": "#FFFFFF",
                                "padding": "20px",
                                "borderRadius": "8px"
                            },
                            children=[
                                # ── header with title + window selector ─────────────────
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "center",
                                        "marginBottom": "10px"
                                    },
                                    children=[
                                        html.H4("Avg. Steps/Questions per Diagnosis",
                                                style={"margin": 0}),
                                        dcc.Dropdown(
                                            id="steps-window-dropdown",
                                            options=[
                                                {"label": "Last 7 days",
                                                    "value": "7d"},
                                                {"label": "Last 30 days",
                                                    "value": "30d"},
                                                {"label": "Last 90 days",
                                                    "value": "90d"},
                                                {"label": "All time",
                                                    "value": "all"},
                                            ],
                                            value="7d",
                                            clearable=False,
                                            style={"width": "150px"}
                                        )
                                    ]
                                ),

                                # ── chart placeholder ────────────────────────────────────
                                dcc.Graph(id="avg-steps-chart",
                                          style={"height": "360px"})
                            ]
                        ),

                        # Completion Rate chart
                        html.Div(
                            style={"flex": "1", "backgroundColor": "#FFFFFF",
                                   "padding": "20px", "borderRadius": "8px"},
                            children=[
                                # ── header row: title + dropdown ─────────────────────────────
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "center",
                                        "marginBottom": "10px"
                                    },
                                    children=[
                                        html.H4("Diagnostic Session Completion Rate",
                                                style={"margin": 0}),
                                        dcc.Dropdown(
                                            id="completion-window-dropdown",
                                            options=[
                                                {"label": "Last 7 days",
                                                    "value": "7d"},
                                                {"label": "Last 30 days",
                                                    "value": "30d"},
                                                {"label": "Last 90 days",
                                                    "value": "90d"},
                                                {"label": "All time",
                                                    "value": "all"},
                                            ],
                                            value="all",
                                            clearable=False,
                                            style={"width": "150px"}
                                        )
                                    ]
                                ),

                                # ── the bar chart ─────────────────────────────────────────────
                                dcc.Graph(id="completion-rate-chart",
                                          style={"height": "360px"})
                            ]
                        ),
                        # Placeholder
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

                # ── Average Diagnosis Time (full width) ─────────────────────
                html.Div(
                    style={"marginTop": "40px", "backgroundColor": "#FFFFFF",
                           "padding": "20px", "borderRadius": "8px"},
                    children=[
                        html.H4("Avg. Diagnosis Time",
                                style={"marginTop": 0}),
                        dcc.Graph(id="avg-diagnosis-time-chart",
                                  style={"height": "360px", "width": "100%"})
                    ]
                ),

                # ── Temporal Metrics ──────────────────────────────────────────
                html.H3("Temporal Metrics", style={
                        "marginTop": "40px", "marginBottom": "20px"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "width": "100%"},
                    children=[
                        # Hourly / Daily Active Users
                        html.Div(
                            style={"flex": "1", "backgroundColor": "#FFFFFF",
                                   "padding": "20px", "borderRadius": "8px"},
                            children=[
                                html.H4("Hourly/Daily Active Users",
                                        style={"marginTop": 0}),
                                dcc.Graph(id="hourly-active-chart",
                                          style={"height": "360px"})
                            ]
                        ),
                        # Most Active Days of Week
                        html.Div(
                            style={"flex": "1", "backgroundColor": "#FFFFFF",
                                   "padding": "20px", "borderRadius": "8px"},
                            children=[
                                html.H4("Most Active Days of the Week",
                                        style={"marginTop": 0}),
                                dcc.Graph(id="weekday-active-chart",
                                          style={"height": "360px"})
                            ]
                        )
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
        Output("top-makes-chart",       "figure"),
        Output("top-models-chart", "figure"),    # ← here
        Output("top-issues-chart",      "figure"),
        Output("total-chats",           "children"),
        Output("avg-steps-chart",       "figure"),
        Output("completion-rate-chart", "figure"),
        Output("avg-diagnosis-time-chart",   "figure"),
        Output("top-parts-chart",            "figure"),
        Output("hourly-active-chart",      "figure"),
        Output("weekday-active-chart",     "figure"),
    ],
    [
        Input("issues-window-dropdown", "value"),
        Input("parts-window-dropdown", "value"),
        Input("steps-window-dropdown", "value"),
        Input("completion-window-dropdown","value"),  # ← here
        Input("interval-component",       "n_intervals")
    ]
)
def update_dashboard(top_issues_window, parts_window, steps_window, comp_window, n):
    # Top 5 makes
    df_makes = fetch_top_makes()
    if df_makes.empty:
        fig_makes = px.bar(title="No data")
    else:
        fig_makes = px.bar(
            df_makes, x="count", y="make", orientation="h",
            labels={"count": "Occurrences", "make": "Make"}, height=360
        )
        fig_makes.update_traces(marker_color="#E8E8E8", marker_line_width=0)
        fig_makes.update_layout(
            yaxis={"autorange": "reversed"}, bargap=0.2,
            margin={"l": 140, "r": 20, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        # right-edge border
        y0, y1 = fig_makes.layout.yaxis.domain
        span = (y1 - y0) / max(len(df_makes), 1)
        for i, row in df_makes.iterrows():
            cnt = row["count"]
            center = y1 - span*(i+0.5)
            half = span*0.4
            fig_makes.add_shape(
                type="line", x0=cnt, x1=cnt,
                y0=center-half, y1=center+half,
                xref="x", yref="paper",
                line=dict(color="#EC6936", width=3)
            )

    # Top 5 models
    df_models = fetch_top_models()
    if df_models.empty:
        fig_models = px.bar(title="No data")
    else:
        fig_models = px.bar(
            df_models, x="count", y="model", orientation="h",
            labels={"count": "Occurrences", "model": "Model"}, height=360
        )
        fig_models.update_traces(marker_color="#E8E8E8", marker_line_width=0)
        fig_models.update_layout(
            yaxis={"autorange": "reversed"}, bargap=0.2,
            margin={"l": 140, "r": 20, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        y0, y1 = fig_models.layout.yaxis.domain
        span = (y1-y0)/max(len(df_models), 1)
        for i, row in df_models.iterrows():
            cnt = row["count"]
            center = y1 - span*(i+0.5)
            half = span*0.4
            fig_models.add_shape(
                type="line", x0=cnt, x1=cnt,
                y0=center-half, y1=center+half,
                xref="x", yref="paper",
                line=dict(color="#EC6936", width=3)
            )

    # Top 5 issues
    df_issues = fetch_top_issues(window=top_issues_window)

    if df_issues.empty:
        fig_issues = px.bar(title="No diagnosed issues yet")
        fig_issues.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig_issues

    fig_issues = px.bar(
        df_issues, x="count", y="issue", orientation="h",
        labels={"count": "Occurrences", "issue": "Issue"},
        height=360
    )
    fig_issues.update_traces(marker_color="#E8E8E8", marker_line_width=0)
    fig_issues.update_layout(
        yaxis={"autorange": "reversed"},
        bargap=0.2,
        margin={"l": 140, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # add the little right-edge orange lines
    y0, y1 = fig_issues.layout.yaxis.domain
    span = (y1 - y0) / max(len(df_issues), 1)
    for i, row in df_issues.iterrows():
        cnt = row["count"]
        center = y1 - span*(i + 0.5)
        half = span * 0.4
        fig_issues.add_shape(
            type="line", x0=cnt, x1=cnt,
            y0=center-half, y1=center+half,
            xref="x", yref="paper",
            line=dict(color="#EC6936", width=3)
        )

    # Total chats
    total_chats = fetch_total_chats()

    # Avg steps chart
# —— Avg steps chart —— #
    df_avg = fetch_avg_steps(window=steps_window)
    if df_avg.empty:
        fig_avg = px.line(title="No data")
        fig_avg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    else:
        fig_avg = px.line(
            df_avg,
            x="day",
            y="avg_steps",
            labels={"day": "Day", "avg_steps": "Avg Steps"},
            height=360
        )
        fig_avg.update_traces(line=dict(color="#EC6936"), marker=dict(size=4))
        fig_avg.update_layout(
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        # now pick tick spacing/format based on window:
        if steps_window == "7d":
            # daily
            fig_avg.update_xaxes(
                tickformat="%a",            # Mon, Tue, …
                dtick=24*3600*1000,         # one day in ms
                ticklabelmode="instant"
            )
        elif steps_window == "30d":
            # weekly ticks
            fig_avg.update_xaxes(
                tickformat="%e %b",         # 27 Apr
                dtick=7*24*3600*1000,       # one week
                ticklabelmode="period",
                tickangle=-45
            )
        elif steps_window == "90d":
            # monthly ticks
            fig_avg.update_xaxes(
                tickformat="%e %b",         # 01 Jun, 01 Jul …
                dtick="M1",                 # one month
                ticklabelmode="period",
                tickangle=-45
            )
        else:  # all time
            # also month-based, but format with year if you like:
            fig_avg.update_xaxes(
                tickformat="%b %Y",         # Apr 2025
                dtick="M1",
                ticklabelmode="period",
                tickangle=-45
            )

    # Completion Rate chart
    df_comp = fetch_completion_rate(window=comp_window)
    if df_comp.empty:
        fig_comp = px.bar(title="No data")
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    else:
        fig_comp = px.bar(
            df_comp, x="status", y="count",
            labels={"status": "Status", "count": "Count"},
            height=360
        )
        fig_comp.update_traces(marker_color="#E8E8E8", marker_line_width=0)
        fig_comp.update_layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        # orange top border
        x0, x1 = fig_comp.layout.xaxis.domain
        span = (x1 - x0) / len(df_comp)
        for i, row in df_comp.iterrows():
            cnt    = row["count"]
            center = x0 + span*(i + 0.5)
            half   = span * 0.4
            fig_comp.add_shape(
                type="line",
                xref="paper", yref="y",
                x0=center-half, x1=center+half,
                y0=cnt, y1=cnt,
                line=dict(color="#EC6936", width=3)
            )

        # ─ Avg diagnosis time by month ──────────────────────────────────────────────
    df_time = fetch_avg_diagnosis_time_by_month()
    if df_time.empty:
        fig_time = px.line(title="No data")
        fig_time.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        fig_time = px.line(df_time, x="month", y="avg_time",
                           labels={"month": "Month", "avg_time": "Avg Mins"}, height=360)
        fig_time.update_traces(line=dict(color="#EC6936"), marker=dict(size=4))
        fig_time.update_layout(margin={"l": 40, "r": 20, "t": 20, "b": 40},
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_time.update_xaxes(dtick="M1", tickformat="%b")

    # Top Selling Parts
    df_parts = fetch_top_parts(window=parts_window)
    if df_parts.empty:
        fig_parts = px.bar(title="No data")
        fig_parts.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    else:
        fig_parts = px.bar(
            df_parts,
            x="count",
            y="part",
            orientation="h",
            labels={"count": "Occurrences", "part": "Part"},
            height=360
        )
        # grey fill, no default border
        fig_parts.update_traces(marker_color="#E8E8E8", marker_line_width=0)
        fig_parts.update_layout(
            yaxis={"autorange": "reversed"},
            bargap=0.2,
            margin={"l": 140, "r": 20, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        # draw the 3px orange edge at the end of each bar
        y0, y1 = fig_parts.layout.yaxis.domain
        span = (y1 - y0) / max(len(df_parts), 1)
        for i, row in df_parts.iterrows():
            cnt = row["count"]
            center = y1 - span * (i + 0.5)
            half = span * 0.4
            fig_parts.add_shape(
                type="line",
                x0=cnt, x1=cnt,
                y0=center - half, y1=center + half,
                xref="x", yref="paper",
                line=dict(color="#EC6936", width=3)
            )

     # Hourly active users
    # a) pull raw GA rows
    df_raw = fetch_hourly_active_users()

    # b) reindex so we have exactly 24 rows, one for each hour
    all_hours = pd.DataFrame({"hour": list(range(24))})
    df = all_hours.merge(df_raw, on="hour", how="left")
    df["users"] = df["users"].fillna(0)

    # c) build human-readable labels
    df["label"] = df["hour"].apply(
        lambda h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}")

    # d) draw the spline chart
    fig_hour = px.line(
        df,
        x="hour",
        y="users",
        custom_data=["label"],
        labels={"hour": "", "users": "Active Users"},
        height=360,
        line_shape="spline"
    )
    fig_hour.update_traces(
        line=dict(color="#EC6936"),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "Active Users=%{y}<extra></extra>"
        )
    )

    # e) only show the four ticks you want
    fig_hour.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 6, 12, 18],
            ticktext=["12 AM", "6 AM", "12 PM", "6 PM"]
        ),
        margin={"t": 20, "b": 30, "l": 40, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # Active by weekday
    df_week = fetch_active_by_weekday()
    if df_week.empty:
        fig_week = px.bar(title="No data")
    else:
        # 1) draw bare bars
        fig_week = px.bar(
            df_week,
            x="day",
            y="users",
            labels={"day": "Day", "users": "Active Users"},
            height=360
        )
        fig_week.update_traces(
            marker_color="#E8E8E8",    # fill
            marker_line_width=0        # no border
        )
        fig_week.update_layout(
            margin={"t": 20, "b": 40, "l": 40, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        # 2) add a thin orange line at the top of each bar
        N = len(df_week)
        for i, row in enumerate(df_week.itertuples()):
            # center of bar i in domain coords
            center = (i + 0.5) / N
            half = 0.4 / N  # bar width is ~0.8 category-width
            y = row.users

            fig_week.add_shape(
                type="line",
                x0=center-half, x1=center+half,
                xref="x domain",
                y0=y, y1=y,
                line=dict(color="#EC6936", width=3)
            )

    return fig_makes, fig_models, fig_issues, total_chats, fig_avg, fig_comp, fig_time, fig_parts, fig_hour, fig_week


# ── Run server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
    # app.run(debug=True)
