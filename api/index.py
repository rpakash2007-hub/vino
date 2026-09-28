import sys
import os
import traceback
from pathlib import Path

# Add project root directory to sys.path so app, config, database, feature_extraction can be imported
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    # Import Flask WSGI application instance
    from app import app

    # Disable debug mode in production serverless environment
    app.debug = False

    # Vercel WSGI entry points (some environments expect `app`, others `handler`)
    handler = app

except Exception as startup_err:
    from flask import Flask, render_template_string
    error_tb = traceback.format_exc()
    print(f"[CRITICAL VERCEL STARTUP ERROR] {startup_err}\n{error_tb}", file=sys.stderr)

    # Fallback diagnostics app to prevent silent 500 FUNCTION_INVOCATION_FAILED
    app = Flask(__name__)
    handler = app

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def vercel_diagnostic(path):
        return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>NephroScan AI - Service Startup Notice</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b1120; color: #f8fafc; padding: 2rem; margin: 0; }
            .box { max-width: 780px; margin: 2rem auto; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            h1 { color: #f43f5e; margin-top: 0; font-size: 1.4rem; }
            pre { background: #030712; padding: 1.25rem; border-radius: 8px; overflow-x: auto; color: #38bdf8; font-size: 0.85rem; line-height: 1.45; border: 1px solid #1f2937; }
            p { color: #94a3b8; font-size: 0.95rem; line-height: 1.6; }
          </style>
        </head>
        <body>
          <div class="box">
            <h1>Application Cold-Start Diagnostic</h1>
            <p>The application encountered an initialization issue during serverless boot:</p>
            <pre>{{ error_tb }}</pre>
          </div>
        </body>
        </html>
        """, error_tb=error_tb), 500

