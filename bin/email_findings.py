import sys
import os
import requests


def send_email(results_path: str):
    api_key = os.environ["RESEND_API_KEY"]

    with open(results_path, "r") as f:
        body = f.read()

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": "FPL Bot <onboarding@resend.dev>",
            "to": os.environ["EMAIL_ADDRESS"],
            "subject": "FPL Weekly Automation Results",
            "text": body,
        },
    )

    response.raise_for_status()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python email_findings.py <results_file>")

    send_email(sys.argv[1])