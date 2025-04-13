# Quick Start Commands

1.  `cd /Users/andrew/Desktop/Projects/main`
2.  `sudo brew services start nginx`
3.  `source venv/bin/activate`
4.  `python slack_hub/slack_app.py`

---

# Running the HTTPS Service

This document outlines the steps to run the Slack application with Nginx reverse proxy and HTTPS enabled via Let's Encrypt/Certbot.

## Prerequisites

*   Nginx installed (via Homebrew: `/opt/homebrew/bin/nginx`) and configured as a reverse proxy for the Flask app (`/opt/homebrew/etc/nginx/servers/slack_app.conf`).
*   Certbot installed and configured for automatic certificate renewal for your domain (`iodx.io`).
*   Python virtual environment (`venv`) in the project root set up with required dependencies (`requirements.txt`).
*   Router port forwarding configured: External TCP ports 80 and 443 forwarded to this machine's internal IP (e.g., 192.168.0.200) on ports 80 and 443 respectively.
*   macOS Firewall (if enabled) allows incoming connections on ports 80 and 443 for Nginx.

## Startup Procedure

Execute these commands in order from the project root directory (`/Users/andrew/Desktop/Projects/main`).

1.  **Start Nginx Service:**
    (Ensures Nginx is running in the background to handle incoming web requests)
    ```bash
    sudo brew services start nginx
    ```
    *(You might need to enter your administrator password)*

2.  **Activate Python Virtual Environment:**
    (Ensures the Flask app uses the correct Python interpreter and dependencies)
    ```bash
    source venv/bin/activate
    ```

3.  **Run the Slack/Flask Application:**
    (Starts the application server, which listens locally for requests proxied by Nginx. This command runs in the foreground and occupies the terminal.)
    ```bash
    python slack_hub/slack_app.py
    ```

The service should now be accessible via `https://iodx.io`. The Flask application logs will appear in the terminal where you ran the last command.

## Stopping the Service

1.  Go to the terminal running the `python slack_hub/slack_app.py` command and press `Ctrl + C`.
2.  Stop the Nginx service:
    ```bash
    sudo brew services stop nginx
    ``` 