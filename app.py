from flask import Flask, render_template, request
import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
from plotly.offline import plot
import json

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
DATA_DIR = config["data_dir"]

warnings.filterwarnings("ignore")

app = Flask(__name__)
os.makedirs("static", exist_ok=True)


def load_model_and_metadata():
    with open(os.path.join(DATA_DIR, config["model_file"]), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(DATA_DIR, config["metadata_file"]), "rb") as f:
        metadata = pickle.load(f)

    return model, metadata


def load_data():
    patients = pd.read_csv(os.path.join(DATA_DIR, config["patients_file"]))
    encounters = pd.read_csv(os.path.join(DATA_DIR, config["encounters_file"]))

    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients["AGE"] = ((pd.Timestamp.today() - patients["BIRTHDATE"]).dt.days / 365.25)

    def make_age_group(age):
        if pd.isna(age):
            return "Unknown"
        elif age <= 18:
            return "0-18"
        elif age <= 40:
            return "19-40"
        elif age <= 65:
            return "41-65"
        else:
            return "65+"

    patients["AGE_GROUP"] = patients["AGE"].apply(make_age_group)

    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
    encounters["STOP"] = pd.to_datetime(encounters["STOP"], errors="coerce")

    encounters["LOS_HOURS"] = (
        (encounters["STOP"] - encounters["START"]).dt.total_seconds() / 3600
    )

    encounters = encounters[encounters["START"].notna()].copy()
    encounters = encounters[encounters["LOS_HOURS"].notna()].copy()
    encounters = encounters[encounters["LOS_HOURS"] >= 0].copy()

    encounters = encounters.merge(
        patients[["Id", "CITY", "AGE_GROUP", "GENDER", "RACE"]],
        left_on="PATIENT",
        right_on="Id",
        how="left"
    )

    return patients, encounters


def get_filter_values(encounters):
    return {
        "cities": sorted(encounters["CITY"].dropna().astype(str).unique().tolist()),
        "encounter_classes": sorted(encounters["ENCOUNTERCLASS"].dropna().astype(str).unique().tolist()),
        "age_groups": sorted(encounters["AGE_GROUP"].dropna().astype(str).unique().tolist()),
        "genders": sorted(encounters["GENDER"].dropna().astype(str).unique().tolist()),
        "races": sorted(encounters["RACE"].dropna().astype(str).unique().tolist())
    }


def apply_filters(df, selected_city, selected_class, selected_age_group, selected_gender, selected_race):
    filtered = df.copy()

    if selected_city != "All":
        filtered = filtered[filtered["CITY"] == selected_city]
    if selected_class != "All":
        filtered = filtered[filtered["ENCOUNTERCLASS"] == selected_class]
    if selected_age_group != "All":
        filtered = filtered[filtered["AGE_GROUP"] == selected_age_group]
    if selected_gender != "All":
        filtered = filtered[filtered["GENDER"] == selected_gender]
    if selected_race != "All":
        filtered = filtered[filtered["RACE"] == selected_race]

    return filtered


def build_monthly_series(filtered):
    monthly = filtered.copy()
    monthly["MONTH"] = monthly["START"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly.groupby("MONTH").size().reset_index(name="TOTAL_ENCOUNTERS")
    monthly = monthly.sort_values("MONTH").reset_index(drop=True)
    return monthly


def build_hourly_series(filtered):
    hourly = filtered.copy()
    hourly["HOUR"] = hourly["START"].dt.floor("h")
    hourly = hourly.groupby("HOUR").size().reset_index(name="TOTAL_ENCOUNTERS")
    hourly = hourly.sort_values("HOUR").reset_index(drop=True)
    return hourly


def build_daily_los_series(filtered):
    los_df = filtered.copy()
    if los_df.empty:
        return pd.DataFrame(columns=["DATE", "AVG_LOS"])

    los_df["DATE_ONLY"] = los_df["START"].dt.floor("D")
    daily_los = (
        los_df.groupby("DATE_ONLY")["LOS_HOURS"]
        .mean()
        .reset_index(name="AVG_LOS")
        .sort_values("DATE_ONLY")
        .reset_index(drop=True)
    )
    return daily_los


def build_los_outlook(filtered):
    daily_los = build_daily_los_series(filtered)

    if daily_los.empty:
        return {
            "current_avg_los": 0,
            "los_outlook_value": 0,
            "los_outlook_label": "Stable LOS",
            "los_guidance": "No LOS trend information is available."
        }

    current_avg_los = round(float(filtered["LOS_HOURS"].mean()), 2) if not filtered.empty else 0

    recent_los = daily_los.tail(3)["AVG_LOS"].mean() if len(daily_los) >= 3 else daily_los["AVG_LOS"].mean()
    baseline_los = daily_los.tail(7)["AVG_LOS"].mean() if len(daily_los) >= 7 else daily_los["AVG_LOS"].mean()

    recent_los = float(recent_los) if pd.notna(recent_los) else 0
    baseline_los = float(baseline_los) if pd.notna(baseline_los) else 0

    change_pct = 0 if baseline_los == 0 else ((recent_los - baseline_los) / baseline_los) * 100
    los_outlook_value = round(recent_los, 2)

    if change_pct > 10:
        los_outlook_label = "Rising LOS Pressure"
        los_guidance = "LOS is rising. Review discharge flow and bed turnover."
    elif change_pct < -10:
        los_outlook_label = "Improving LOS"
        los_guidance = "LOS is improving. Throughput is more stable than the recent baseline."
    else:
        los_outlook_label = "Stable LOS"
        los_guidance = "LOS is relatively stable compared with the recent baseline."

    return {
        "current_avg_los": round(current_avg_los, 2),
        "los_outlook_value": los_outlook_value,
        "los_outlook_label": los_outlook_label,
        "los_guidance": los_guidance
    }


def save_monthly_chart(monthly):
    plt.figure(figsize=(10, 4.4))

    if not monthly.empty:
        recent_monthly = monthly.tail(12)
        plt.plot(recent_monthly["MONTH"], recent_monthly["TOTAL_ENCOUNTERS"], linewidth=2)

    plt.title("Monthly Encounter Trend", fontsize=15)
    plt.xlabel("Month")
    plt.ylabel("Encounters")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("static/monthly_encounters.png")
    plt.close()


def save_top_cities_chart(filtered):
    city_counts = filtered.groupby("CITY").size().reset_index(name="TOTAL_ENCOUNTERS")
    city_counts = city_counts.sort_values("TOTAL_ENCOUNTERS", ascending=False).head(8)

    plt.figure(figsize=(9, 4.4))
    if not city_counts.empty:
        plt.barh(city_counts["CITY"], city_counts["TOTAL_ENCOUNTERS"])
        plt.gca().invert_yaxis()

    plt.title("Top Cities by Encounter Volume", fontsize=15)
    plt.xlabel("Total Encounters")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig("static/top_cities.png")
    plt.close()


def build_forecast_plot(forecast_df):
    if forecast_df.empty:
        return "<p>No forecast available.</p>"

    df = forecast_df.copy()

    def get_block(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    df["HOUR_NUM"] = df["DATE"].dt.hour
    df["DAY"] = df["DATE"].dt.strftime("%b %d")
    df["BLOCK"] = df["HOUR_NUM"].apply(get_block)
    df["LABEL"] = df["DAY"] + " - " + df["BLOCK"]

    grouped = df.groupby("LABEL", as_index=False)["PREDICTED_ENCOUNTERS"].mean()
    grouped = grouped.head(8)

    if grouped.empty:
        return "<p>No forecast available.</p>"

    grouped["PREDICTED_ENCOUNTERS"] = grouped["PREDICTED_ENCOUNTERS"].round(1)
    max_idx = grouped["PREDICTED_ENCOUNTERS"].idxmax()

    colors = ["#9ec5db"] * len(grouped)
    if len(grouped) > 0:
        colors[max_idx] = "#d9534f"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped["LABEL"],
        y=grouped["PREDICTED_ENCOUNTERS"],
        marker_color=colors,
        text=grouped["PREDICTED_ENCOUNTERS"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Expected Patients: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Demand Forecast (Next 72 Hours)",
        xaxis_title="Upcoming Time Blocks",
        yaxis_title="Expected Patients",
        template="plotly_white",
        height=370,
        showlegend=False,
        margin=dict(l=25, r=20, t=55, b=75)
    )

    fig.update_xaxes(tickangle=-18)
    fig.update_yaxes(showgrid=True)

    return plot(
        fig,
        output_type="div",
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True}
    )


def predict_next_72_hours(model, hourly):
    if hourly.empty or len(hourly) < 24:
        future_hours = pd.date_range(
            start=pd.Timestamp.now().floor("h"),
            periods=72,
            freq="h"
        )
        return pd.DataFrame({
            "DATE": future_hours,
            "PREDICTED_ENCOUNTERS": [0] * 72
        })

    history = hourly.copy()
    predictions = []

    for _ in range(72):
        next_hour = history["HOUR"].max() + pd.Timedelta(hours=1)

        lag_1 = history["TOTAL_ENCOUNTERS"].iloc[-1]
        lag_2 = history["TOTAL_ENCOUNTERS"].iloc[-2]
        lag_3 = history["TOTAL_ENCOUNTERS"].iloc[-3]
        lag_24 = history["TOTAL_ENCOUNTERS"].iloc[-24] if len(history) >= 24 else history["TOTAL_ENCOUNTERS"].mean()

        rolling_mean_6 = history["TOTAL_ENCOUNTERS"].tail(6).mean()
        rolling_mean_24 = history["TOTAL_ENCOUNTERS"].tail(24).mean()

        row = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_24": lag_24,
            "rolling_mean_6": rolling_mean_6,
            "rolling_mean_24": rolling_mean_24,
            "hour_of_day": next_hour.hour,
            "day_of_week": next_hour.dayofweek
        }])

        pred = float(model.predict(row)[0])
        pred = max(round(pred, 2), 0)

        predictions.append({
            "DATE": next_hour,
            "PREDICTED_ENCOUNTERS": pred
        })

        history = pd.concat([
            history,
            pd.DataFrame({
                "HOUR": [next_hour],
                "TOTAL_ENCOUNTERS": [pred]
            })
        ], ignore_index=True)

    return pd.DataFrame(predictions)


def get_top_encounter_classes(filtered, top_n=3):
    if filtered.empty or "ENCOUNTERCLASS" not in filtered.columns:
        return []

    top_classes = (
        filtered["ENCOUNTERCLASS"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )
    return top_classes


def build_encounter_class_forecasts(model, filtered, top_n=3):
    forecast_rows = []
    top_classes = get_top_encounter_classes(filtered, top_n=top_n)

    for encounter_class in top_classes:
        class_df = filtered[filtered["ENCOUNTERCLASS"] == encounter_class].copy()
        class_hourly = build_hourly_series(class_df)

        if class_hourly.empty or len(class_hourly) < 24:
            peak_value = 0
            peak_time = "N/A"
        else:
            class_forecast = predict_next_72_hours(model, class_hourly)
            if class_forecast.empty:
                peak_value = 0
                peak_time = "N/A"
            else:
                peak_row = class_forecast.loc[class_forecast["PREDICTED_ENCOUNTERS"].idxmax()]
                peak_value = round(float(peak_row["PREDICTED_ENCOUNTERS"]), 2)
                peak_time = peak_row["DATE"].strftime("%b %d, %I:%M %p")

        forecast_rows.append({
            "encounter_class": encounter_class,
            "peak_value": peak_value,
            "peak_time": peak_time
        })

    return forecast_rows


def get_recent_trend_summary(filtered):
    if filtered.empty:
        return "No recent trend information is available."

    latest_time = filtered["START"].max()

    current_24h_start = latest_time - pd.Timedelta(hours=24)
    previous_24h_start = latest_time - pd.Timedelta(hours=48)

    current_24h = filtered[filtered["START"] >= current_24h_start]
    previous_24h = filtered[
        (filtered["START"] >= previous_24h_start) &
        (filtered["START"] < current_24h_start)
    ]

    current_count = len(current_24h)
    previous_count = len(previous_24h)

    if previous_count == 0:
        return f"{current_count} encounters were recorded in the last 24 hours."

    change_pct = ((current_count - previous_count) / previous_count) * 100

    if change_pct > 10:
        return f"Demand is rising. The last 24 hours are up {change_pct:.1f}% versus the prior 24 hours."
    elif change_pct < -10:
        return f"Demand is easing. The last 24 hours are down {abs(change_pct):.1f}% versus the prior 24 hours."
    else:
        return "Demand is relatively stable compared with the previous 24 hours."


def build_ai_operations_insight(
    risk_level,
    peak_forecast_value,
    peak_forecast_hour,
    los_info,
    recent_trend_summary,
    encounter_class_forecasts,
    recent_24h_encounters,
    recent_7d_encounters
):
    top_class_text = "No dominant encounter class identified."
    if encounter_class_forecasts:
        top_class = max(encounter_class_forecasts, key=lambda x: x["peak_value"])
        top_class_text = (
            f"{top_class['encounter_class']} is expected to have the highest class-level peak "
            f"at {top_class['peak_value']} encounters around {top_class['peak_time']}."
        )

    recent_activity_ratio = recent_24h_encounters / recent_7d_encounters if recent_7d_encounters > 0 else 0

    if risk_level == "High Risk":
        ai_priority = "Immediate Operational Review"
        ai_priority_class = "priority-high"
        ai_summary = (
            f"High demand pressure is expected. The forecast peak is {peak_forecast_value} patients "
            f"around {peak_forecast_hour}. {top_class_text}"
        )
        ai_recommendation = (
            "Review staffing coverage, bed availability, and discharge readiness before the projected peak window. "
            "Prioritize units linked to the highest expected encounter class."
        )
    elif risk_level == "Medium Risk":
        ai_priority = "Monitor and Prepare"
        ai_priority_class = "priority-medium"
        ai_summary = (
            f"Moderate operational pressure is expected. The forecast peak is {peak_forecast_value} patients "
            f"around {peak_forecast_hour}. {top_class_text}"
        )
        ai_recommendation = (
            "Monitor patient flow, confirm staffing flexibility, and prepare contingency coverage if demand rises."
        )
    else:
        ai_priority = "Normal Monitoring"
        ai_priority_class = "priority-low"
        ai_summary = (
            f"Demand appears stable. The expected peak is {peak_forecast_value} patients "
            f"around {peak_forecast_hour}. {top_class_text}"
        )
        ai_recommendation = (
            "Maintain normal staffing and continue routine monitoring of demand and length-of-stay trends."
        )

    if los_info["los_outlook_label"] == "Rising LOS Pressure":
        ai_recommendation += (
            " Length of stay is also rising, so discharge coordination and bed turnover should be reviewed."
        )

    if recent_activity_ratio > 0.35:
        ai_summary += (
            " A relatively high share of recent weekly encounters occurred in the last 24 hours, suggesting short-term demand concentration."
        )

    if risk_level == "High Risk":
        confidence_label = "Moderate Confidence"
        confidence_note = (
            "Forecast direction is useful for planning, but high-demand periods should be reviewed with operational context."
        )
    elif recent_7d_encounters < 25:
        confidence_label = "Low Confidence"
        confidence_note = (
            "The selected filter group has limited recent data, so the forecast should be interpreted cautiously."
        )
    else:
        confidence_label = "High Confidence"
        confidence_note = (
            "The selected view has enough recent activity for a stable planning-level forecast."
        )

    return {
        "ai_priority": ai_priority,
        "ai_priority_class": ai_priority_class,
        "ai_summary": ai_summary,
        "ai_recommendation": ai_recommendation,
        "confidence_label": confidence_label,
        "confidence_note": confidence_note
    }

def build_scenario_simulator(
    selected_scenario,
    peak_forecast_value,
    peak_forecast_hour,
    los_info
):
    scenario_map = {
        "normal": {
            "label": "Normal Demand",
            "multiplier": 1.00,
            "description": "Baseline forecast with no additional demand pressure."
        },
        "surge_10": {
            "label": "+10% Demand Surge",
            "multiplier": 1.10,
            "description": "Moderate increase in expected patient demand."
        },
        "surge_20": {
            "label": "+20% Demand Surge",
            "multiplier": 1.20,
            "description": "Noticeable increase in demand requiring closer monitoring."
        },
        "surge_35": {
            "label": "+35% Demand Surge",
            "multiplier": 1.35,
            "description": "High surge scenario requiring proactive capacity preparation."
        },
        "high_los": {
            "label": "High LOS Pressure",
            "multiplier": 1.15,
            "description": "Simulates slower bed turnover due to longer stays."
        }
    }

    scenario = scenario_map.get(selected_scenario, scenario_map["normal"])
    simulated_peak = round(peak_forecast_value * scenario["multiplier"], 2)

    if simulated_peak < 5:
        simulated_risk = "Low Risk"
        simulated_class = "risk-low"
        simulated_action = "Maintain normal staffing and continue routine monitoring."
    elif simulated_peak <= 15:
        simulated_risk = "Medium Risk"
        simulated_class = "risk-medium"
        simulated_action = "Prepare flexible staffing and monitor bed turnover during the peak window."
    else:
        simulated_risk = "High Risk"
        simulated_class = "risk-high"
        simulated_action = "Prepare surge staffing, review bed availability, and coordinate discharge readiness."

    if selected_scenario == "high_los" or los_info["los_outlook_label"] == "Rising LOS Pressure":
        simulated_action += " Prioritize discharge planning because LOS pressure may reduce available capacity."

    return {
        "scenario_options": scenario_map,
        "selected_scenario": selected_scenario,
        "scenario_label": scenario["label"],
        "scenario_description": scenario["description"],
        "simulated_peak": simulated_peak,
        "simulated_peak_hour": peak_forecast_hour,
        "simulated_risk": simulated_risk,
        "simulated_class": simulated_class,
        "simulated_action": simulated_action
    }


def build_executive_brief_and_checklist(
    risk_level,
    peak_forecast_value,
    peak_forecast_hour,
    los_info,
    recent_24h_encounters,
    recent_7d_encounters
):
    if risk_level == "High Risk":
        brief_title = "High-Priority Capacity Brief"
        brief_summary = (
            f"High demand is expected with a forecast peak of {peak_forecast_value} patients "
            f"around {peak_forecast_hour}. Operations should prepare for capacity pressure."
        )
        checklist = [
            "Review staffing coverage for the projected peak window",
            "Check bed availability and discharge readiness",
            "Coordinate with ER, inpatient, and support teams",
            "Monitor demand changes after the next model refresh"
        ]

    elif risk_level == "Medium Risk":
        brief_title = "Moderate Capacity Brief"
        brief_summary = (
            f"Moderate demand is expected with a forecast peak of {peak_forecast_value} patients "
            f"around {peak_forecast_hour}. Capacity should be monitored closely."
        )
        checklist = [
            "Confirm flexible staffing coverage",
            "Monitor patient flow and length-of-stay trends",
            "Prepare contingency coverage if demand increases",
            "Review upcoming high-demand time blocks"
        ]

    else:
        brief_title = "Stable Operations Brief"
        brief_summary = (
            f"Demand appears stable with a forecast peak of {peak_forecast_value} patients "
            f"around {peak_forecast_hour}. Normal monitoring is appropriate."
        )
        checklist = [
            "Maintain normal staffing levels",
            "Continue routine demand monitoring",
            "Review LOS trends for early warning signals",
            "Reassess after the next scheduled model refresh"
        ]

    if los_info["los_outlook_label"] == "Rising LOS Pressure":
        checklist.append("Prioritize discharge planning due to rising LOS pressure")

    recent_share = round((recent_24h_encounters / recent_7d_encounters) * 100, 1) if recent_7d_encounters > 0 else 0

    brief_metrics = [
        {"label": "Forecast Peak", "value": peak_forecast_value},
        {"label": "Peak Time", "value": peak_forecast_hour},
        {"label": "LOS Status", "value": los_info["los_outlook_label"]},
        {"label": "24h Share of 7d Activity", "value": f"{recent_share}%"}
    ]

    return {
        "brief_title": brief_title,
        "brief_summary": brief_summary,
        "brief_metrics": brief_metrics,
        "action_checklist": checklist
    }


@app.route("/")
def home():
    patients, encounters = load_data()
    filter_values = get_filter_values(encounters)

    selected_city = request.args.get("city", "All")
    selected_class = request.args.get("encounter_class", "All")
    selected_age_group = request.args.get("age_group", "All")
    selected_gender = request.args.get("gender", "All")
    selected_race = request.args.get("race", "All")

    filtered = apply_filters(
        encounters,
        selected_city,
        selected_class,
        selected_age_group,
        selected_gender,
        selected_race
    )

    total_patients = filtered["PATIENT"].nunique() if not filtered.empty else 0
    total_encounters = len(filtered)
    avg_los = round(filtered["LOS_HOURS"].mean(), 2) if not filtered.empty else 0

    recent_24h_encounters = 0
    recent_7d_encounters = 0
    recent_busiest_hour = "N/A"

    if not filtered.empty:
        latest_time = filtered["START"].max()

        recent_24h = filtered[filtered["START"] >= (latest_time - pd.Timedelta(hours=24))]
        recent_7d = filtered[filtered["START"] >= (latest_time - pd.Timedelta(days=7))]

        recent_24h_encounters = len(recent_24h)
        recent_7d_encounters = len(recent_7d)

        recent_hourly = build_hourly_series(recent_7d)
        if not recent_hourly.empty:
            busiest_row = recent_hourly.loc[recent_hourly["TOTAL_ENCOUNTERS"].idxmax()]
            recent_busiest_hour = busiest_row["HOUR"].strftime("%b %d, %I:%M %p")

    monthly = build_monthly_series(filtered)
    hourly = build_hourly_series(filtered)
    model, metadata = load_model_and_metadata()

    forecast_df = predict_next_72_hours(model, hourly)
    los_info = build_los_outlook(filtered)
    encounter_class_forecasts = build_encounter_class_forecasts(model, filtered, top_n=3)

    save_monthly_chart(monthly)
    save_top_cities_chart(filtered)
    forecast_plot = build_forecast_plot(forecast_df)

    forecast_preview = forecast_df.head(6).copy()
    forecast_values = [
        {
            "month": row["DATE"].strftime("%b %d, %I:%M %p"),
            "value": round(float(row["PREDICTED_ENCOUNTERS"]), 2)
        }
        for _, row in forecast_preview.iterrows()
    ]

    if not forecast_df.empty:
        peak_row = forecast_df.loc[forecast_df["PREDICTED_ENCOUNTERS"].idxmax()]
        peak_forecast_hour = peak_row["DATE"].strftime("%b %d, %I:%M %p")
        peak_forecast_value = round(float(peak_row["PREDICTED_ENCOUNTERS"]), 2)
    else:
        peak_forecast_hour = "N/A"
        peak_forecast_value = 0

    forecast_horizon = "Next 72 Hours"

    if peak_forecast_value < 5:
        risk_level = "Low Risk"
        risk_class = "risk-low"
    elif peak_forecast_value <= 15:
        risk_level = "Medium Risk"
        risk_class = "risk-medium"
    else:
        risk_level = "High Risk"
        risk_class = "risk-high"

    if risk_level == "High Risk" and los_info["los_outlook_label"] == "Rising LOS Pressure":
        insight = (
            f"High demand is expected, with a peak of {peak_forecast_value} around {peak_forecast_hour}. "
            f"LOS is also rising, which may increase bed pressure."
        )
        recommended_action = (
            "Increase staffing readiness, review bed availability, and investigate discharge bottlenecks before the projected peak."
        )
    elif risk_level == "High Risk":
        insight = (
            f"High demand is expected, with a peak of {peak_forecast_value} around {peak_forecast_hour}. "
            f"This may place pressure on staffing and patient flow."
        )
        recommended_action = "Prepare staffing and bed plans for the projected peak window."
    elif risk_level == "Medium Risk" and los_info["los_outlook_label"] == "Rising LOS Pressure":
        insight = (
            f"Moderate demand is expected, with a peak of {peak_forecast_value} around {peak_forecast_hour}. "
            f"Rising LOS may tighten capacity."
        )
        recommended_action = "Review shift flexibility and monitor throughput closely."
    elif risk_level == "Medium Risk":
        insight = (
            f"Moderate demand is expected, with a peak of {peak_forecast_value} around {peak_forecast_hour}. "
            f"Capacity may tighten during peak periods."
        )
        recommended_action = "Monitor staffing and bed turnover and prepare for a short-term increase in patient load."
    else:
        insight = (
            f"Demand is expected to remain stable, with a peak of {peak_forecast_value} around {peak_forecast_hour}. "
            f"Current capacity appears sufficient."
        )
        recommended_action = "Maintain normal staffing levels and continue routine monitoring."

    recent_trend_summary = get_recent_trend_summary(filtered)

    ai_ops = build_ai_operations_insight(
        risk_level=risk_level,
        peak_forecast_value=peak_forecast_value,
        peak_forecast_hour=peak_forecast_hour,
        los_info=los_info,
        recent_trend_summary=recent_trend_summary,
        encounter_class_forecasts=encounter_class_forecasts,
        recent_24h_encounters=recent_24h_encounters,
        recent_7d_encounters=recent_7d_encounters
    )

    executive_brief = build_executive_brief_and_checklist(
        risk_level=risk_level,
        peak_forecast_value=peak_forecast_value,
        peak_forecast_hour=peak_forecast_hour,
        los_info=los_info,
        recent_24h_encounters=recent_24h_encounters,
        recent_7d_encounters=recent_7d_encounters
    )

    forecast_explanation = (
        "This forecast uses recent hourly demand, previous-day patterns, and rolling averages to estimate patient load over the next 72 hours."
    )

    validation_note = (
        "Forecasts are generated using the same feature-engineering pipeline each time the app runs, supporting consistent planning-level results when newer data is loaded."
    )

    limitation_note = (
        "Predictions are based on historical patterns from synthetic healthcare data. Unexpected operational shocks or very small filtered groups may reduce forecast stability."
    )

    model_type = "Regression (Lag-Based Time Series)"
    validation_method = "Historical holdout evaluation using MAE and RMSE"
    model_status = "Scheduled refresh enabled"

    last_refresh = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    if "last_trained_hour" in metadata:
        model_last_trained = pd.to_datetime(metadata["last_trained_hour"]).strftime("%Y-%m-%d %H:%M:%S")
    else:
        model_last_trained = "Not Available"

    return render_template(
        "index.html",
        cities=filter_values["cities"],
        encounter_classes=filter_values["encounter_classes"],
        age_groups=filter_values["age_groups"],
        genders=filter_values["genders"],
        races=filter_values["races"],
        selected_city=selected_city,
        selected_class=selected_class,
        selected_age_group=selected_age_group,
        selected_gender=selected_gender,
        selected_race=selected_race,
        recent_24h_encounters=recent_24h_encounters,
        recent_7d_encounters=recent_7d_encounters,
        recent_busiest_hour=recent_busiest_hour,
        total_patients=total_patients,
        total_encounters=total_encounters,
        avg_los=avg_los,
        current_avg_los=los_info["current_avg_los"],
        los_outlook_value=los_info["los_outlook_value"],
        los_outlook_label=los_info["los_outlook_label"],
        los_guidance=los_info["los_guidance"],
        encounter_class_forecasts=encounter_class_forecasts,
        forecast_values=forecast_values,
        forecast_plot=forecast_plot,
        peak_forecast_hour=peak_forecast_hour,
        peak_forecast_value=peak_forecast_value,
        forecast_horizon=forecast_horizon,
        risk_level=risk_level,
        risk_class=risk_class,
        insight=insight,
        recommended_action=recommended_action,
        recent_trend_summary=recent_trend_summary,
        ai_priority=ai_ops["ai_priority"],
        ai_priority_class=ai_ops["ai_priority_class"],
        ai_summary=ai_ops["ai_summary"],
        ai_recommendation=ai_ops["ai_recommendation"],
        confidence_label=ai_ops["confidence_label"],
        confidence_note=ai_ops["confidence_note"],
        brief_title=executive_brief["brief_title"],
        brief_summary=executive_brief["brief_summary"],
        brief_metrics=executive_brief["brief_metrics"],
        action_checklist=executive_brief["action_checklist"],
        forecast_explanation=forecast_explanation,
        validation_note=validation_note,
        limitation_note=limitation_note,
        model_type=model_type,
        validation_method=validation_method,
        model_status=model_status,
        last_refresh=last_refresh,
        model_last_trained=model_last_trained
    )


if __name__ == "__main__":
    app.run(debug=True)