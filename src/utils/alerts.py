import os
import smtplib
from email.mime.text import MIMEText
import requests

# --- CONFIGURATION ---
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")  # Set this environment variable
EMAIL_ALERTS_TO = os.environ.get("ALERTS_EMAIL_TO")      # Set recipient
EMAIL_ALERTS_FROM = os.environ.get("ALERTS_EMAIL_FROM")  # Set sender
SMTP_SERVER = os.environ.get("SMTP_SERVER")              # SMTP server (for email alerts)

# --- ALERT ENGINE ---
def send_slack_alert(message):
    """Send a Slack alert via incoming webhook."""
    if not SLACK_WEBHOOK_URL:
        print("[ALERT] Slack webhook not configured.")
        return False
    return requests.post(SLACK_WEBHOOK_URL, json={"text": message}).ok

def send_email_alert(message, subject="Performance Alert"):
    """Send an email alert."""
    if not (EMAIL_ALERTS_TO and EMAIL_ALERTS_FROM and SMTP_SERVER):
        print("[ALERT] Email configuration not set.")
        return False
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ALERTS_FROM
    msg["To"] = EMAIL_ALERTS_TO
    try:
        with smtplib.SMTP(SMTP_SERVER) as server:
            server.sendmail(EMAIL_ALERTS_FROM, [EMAIL_ALERTS_TO], msg.as_string())
        return True
    except Exception as e:
        print(f"[ALERT ERROR] {e}")
        return False

def trigger_performance_alert(kind, value, target, context=None):
    msg = f"[ALERT] Performance degraded: {kind} = {value} (target: {target})"
    if context:
        msg += f" | Context: {context}"
    print(msg)
    send_slack_alert(msg)
    send_email_alert(msg)

