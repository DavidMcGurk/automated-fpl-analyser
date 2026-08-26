import html
import os
import sys
from datetime import datetime
from typing import Any


def _parse_result(path: str) -> dict[str, Any]:
    """Parse the plain-text email content file written by run.py into a dict."""
    with open(path) as f:
        text = f.read()

    result: dict[str, Any] = {
        "user_id": None,
        "current_squad_xp": 0.0,
        "optimised_squad_xp": 0.0,
        "transfers_used": 0,
        "point_hit": 0,
        "net_improvement": 0.0,
        "suggestions": [],
        "no_transfers": False,
    }

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Team Optimisation Results for User "):
            result["user_id"] = line.split("User ")[-1]
        elif line.startswith("Current squad xP:"):
            result["current_squad_xp"] = float(line.split(":")[1].strip())
        elif line.startswith("Optimised squad xP:"):
            result["optimised_squad_xp"] = float(line.split(":")[1].strip())
        elif line.startswith("Transfers used:"):
            result["transfers_used"] = int(line.split(":")[1].strip())
        elif line.startswith("Point hit:"):
            result["point_hit"] = int(line.split(":")[1].strip())
        elif line.startswith("Net improvement:"):
            result["net_improvement"] = float(line.split(":")[1].strip().replace("+", ""))
        elif line.startswith("No beneficial transfers found."):
            result["no_transfers"] = True

    # Parse suggestions
    lines = text.splitlines()
    in_suggestions = False
    current_suggestion: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        if stripped == "Suggested transfers:":
            in_suggestions = True
            continue
        if not in_suggestions:
            continue
        if stripped.startswith("=" * 10) or stripped == "No beneficial transfers found.":
            break
        if stripped and " -> " in stripped and "xP gain:" not in stripped:
            if current_suggestion:
                result["suggestions"].append(current_suggestion)
            parts = stripped.split(" -> ")
            out_part = parts[0]
            in_part = parts[1] if len(parts) > 1 else ""
            current_suggestion = {"player_out": out_part, "player_in": in_part, "details": ""}
        elif stripped.startswith("xP gain:"):
            current_suggestion["details"] = stripped
            result["suggestions"].append(current_suggestion)
            current_suggestion = {}

    if current_suggestion and current_suggestion.get("player_out"):
        result["suggestions"].append(current_suggestion)

    return result


def _build_html_email(result: dict[str, Any]) -> str:
    """Build an HTML email from the parsed optimisation result."""
    user_id = html.escape(str(result.get("user_id", "Unknown")))
    current_xp = result["current_squad_xp"]
    optimised_xp = result["optimised_squad_xp"]
    transfers_used = result["transfers_used"]
    point_hit = result["point_hit"]
    net_improvement = result["net_improvement"]
    suggestions = result["suggestions"]
    no_transfers = result["no_transfers"]

    date_str = datetime.utcnow().strftime("%A, %d %B %Y")

    improvement_color = "#28a745" if net_improvement > 0 else "#6c757d"
    improvement_sign = "+" if net_improvement > 0 else ""

    suggestions_html = ""
    if no_transfers or not suggestions:
        suggestions_html = """
        <tr>
          <td colspan="3" style="padding: 16px; text-align: center; color: #6c757d;">
            No beneficial transfers found. Your current squad is already optimised!
          </td>
        </tr>"""
    else:
        for s in suggestions:
            player_out = html.escape(s.get("player_out", ""))
            player_in = html.escape(s.get("player_in", ""))
            details = html.escape(s.get("details", ""))
            suggestions_html += f"""
        <tr>
          <td style="padding: 10px 16px; color: #dc3545;">&#10060; {player_out}</td>
          <td style="padding: 10px 16px; color: #28a745;">&#9989; {player_in}</td>
          <td style="padding: 10px 16px; color: #6c757d; font-size: 13px;">{details}</td>
        </tr>"""

    return f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <meta name="supported-color-schemes" content="light dark">
  <style>
    /* Dark mode overrides: keep header text white on dark purple background */
    @media (prefers-color-scheme: dark) {{
      .header-title, .header-sub, .header-id {{
        color: #ffffff !important;
      }}
    }}
  </style>
</head>
<body style="margin: 0; padding: 0; background-color: #f4f4f4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f4f4f4; padding: 24px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">

          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #37003c 0%, #53005a 100%); padding: 32px 40px;">
              <h1 class="header-title" style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 700;">&#9917; FPL Weekly Recommendations</h1>
              <p class="header-sub" style="margin: 8px 0 0; color: #ffffff; opacity: 0.8; font-size: 14px;">{date_str}</p>
              <p class="header-id" style="margin: 4px 0 0; color: #ffffff; opacity: 0.6; font-size: 13px;">Team ID: {user_id}</p>
            </td>
          </tr>

          <!-- Summary stats -->
          <tr>
            <td style="padding: 32px 40px 0;">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td width="50%" style="padding: 16px; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
                    <p style="margin: 0; color: #6c757d; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Current Squad <span style="text-transform: lowercase;">x</span>P</p>
                    <p style="margin: 4px 0 0; font-size: 28px; font-weight: 700; color: #37003c;">{current_xp:.2f}</p>
                  </td>
                  <td width="50%" style="padding: 16px; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
                    <p style="margin: 0; color: #6c757d; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Optimised Squad <span style="text-transform: lowercase;">x</span>P</p>
                    <p style="margin: 4px 0 0; font-size: 28px; font-weight: 700; color: #37003c;">{optimised_xp:.2f}</p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Net improvement -->
          <tr>
            <td style="padding: 16px 40px 0;">
              <div style="background-color: #f0fff4; border: 1px solid #d4edda; border-radius: 8px; padding: 16px; text-align: center;">
                <span style="font-size: 16px; font-weight: 600; color: {improvement_color};">
                  Net Improvement: {improvement_sign}{net_improvement:.2f} pts
                </span>
                <span style="font-size: 13px; color: #6c757d; margin-left: 12px;">
                  ({transfers_used} transfer{'s' if transfers_used != 1 else ''} used, {point_hit}pt hit)
                </span>
              </div>
            </td>
          </tr>

          <!-- Transfer suggestions -->
          <tr>
            <td style="padding: 24px 40px;">
              <h2 style="margin: 0 0 16px; font-size: 18px; color: #37003c;">Transfer Suggestions</h2>
              <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; border: 1px solid #e9ecef; border-radius: 8px; overflow: hidden;">
                <thead>
                  <tr style="background-color: #f8f9fa;">
                    <th style="padding: 10px 16px; text-align: left; font-size: 13px; color: #495057;">Out</th>
                    <th style="padding: 10px 16px; text-align: left; font-size: 13px; color: #495057;">In</th>
                    <th style="padding: 10px 16px; text-align: left; font-size: 13px; color: #495057;">Details</th>
                  </tr>
                </thead>
                <tbody>{suggestions_html}
                </tbody>
              </table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding: 24px 40px 32px; border-top: 1px solid #e9ecef;">
              <p style="margin: 0; font-size: 12px; color: #6c757d; text-align: center;">
                Generated by automated-fpl-analyser &middot; Training logs available in GitHub Actions
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def send_email(results_path: str):
    import requests

    api_key = os.environ["RESEND_API_KEY"]

    result = _parse_result(results_path)
    html_body = _build_html_email(result)

    has_transfers = bool(result.get("suggestions")) and not result.get("no_transfers")
    subject = f"FPL Recommendations - {'Transfers available' if has_transfers else 'No transfers needed'}"

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": "FPL Bot <onboarding@resend.dev>",
            "to": os.environ["EMAIL_ADDRESS"],
            "subject": subject,
            "html": html_body,
        },
    )

    response.raise_for_status()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python email_findings.py <results_file>")

    send_email(sys.argv[1])
