import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import time
import re
import json
import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Attempt to import Google Drive API client
try:
    from googleapiclient.discovery import build
except ModuleNotFoundError:
    build = None

# Environment configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
TEMPLATE_SHEET_ID = os.getenv(
    "TEMPLATE_SHEET_ID",
    "1WOozcrnam_fdNsqKhq8E517J5nI076MsmPIYiGSs2ro"
)

# Initialize OpenAI client
client: OpenAI

def init_openai():
    """Configure the OpenAI client using the API key from environment."""
    global client
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in environment.")
        st.stop()
    client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Google Sheets & Drive clients
gs_client = None
drive_service = None

def init_gs():
    """Authorize gspread and Drive using the service account JSON file."""
    global gs_client, drive_service
    if gs_client is not None:
        return
    try:
        creds_dict = json.load(open(SERVICE_ACCOUNT_FILE))
    except Exception:
        st.error(f"Could not read service account file at {SERVICE_ACCOUNT_FILE}")
        st.stop()
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gs_client = gspread.authorize(creds)
    if build:
        drive_service = build('drive', 'v3', credentials=creds)

# Utility: create a fresh copy of the template spreadsheet

def create_new_sheet():
    """Copy the template sheet, make it public/editable, and return the new spreadsheet ID."""
    init_gs()
    if not TEMPLATE_SHEET_ID:
        st.error("Missing TEMPLATE_SHEET_ID in environment.")
        st.stop()
    if not build or not drive_service:
        st.error("google-api-python-client is required to create a sheet. Install with `pip install google-api-python-client`.")
        st.stop()

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    copy_title = f"Product Optimizer {now}"

    new_file = drive_service.files().copy(
        fileId=TEMPLATE_SHEET_ID,
        body={"name": copy_title}
    ).execute()

    # Make the sheet publicly editable
    drive_service.permissions().create(
        fileId=new_file['id'],
        body={
            'role': 'writer',
            'type': 'anyone'
        },
        fields='id'
    ).execute()

    public_url = f"https://docs.google.com/spreadsheets/d/{new_file['id']}"
    st.success(f"✅ Sheet created and shared: [Open Sheet]({public_url})")
    return new_file.get('id')

# Core OpenAI call with manual retries
def call_openai_with_retries(prompt: str, max_retries: int = 3, backoff: float = 10.0):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
            else:
                st.error(f"OpenAI request failed: {e}")
                st.stop()

# Fetch PDP page text
def fetch_pdp_text(url: str, max_chars: int = 3000) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:max_chars]
    except Exception:
        return ""

# Build a batch prompt
def build_batch_prompt(batch: list[dict]) -> str:
    prompt = (
        "You are a Google Shopping product feed optimization expert.\n"
        "Generate a concise, keyword-rich title under 150 characters for each product. "
        "Include brand, product type, attributes, and use case. Avoid promotional language. "
        "Return only numbered titles.\n\n"
    )
    for idx, item in enumerate(batch, start=1):
        prompt += f"Product {idx}:\n"
        prompt += f"- Original Title: {item['Original Title']}\n"
        if item.get('Product Description'):
            prompt += f"- Product Description: {item['Product Description']}\n"
        prompt += f"- Product Type: {item['Product Type']}\n"
        prompt += f"- Key Attributes: {item['Key Attributes']}\n"
        prompt += f"- Use Case: {item['Use Case']}\n"
        prompt += f"- PDP Page Content: {item['PDP Content']}\n\n"
    prompt += "Return:\n"
    for i in range(1, len(batch) + 1):
        prompt += f"{i}. \n"
    return prompt

# Read & write Google Sheet
def sheet_to_df(sheet_id: str, ws_name: str):
    init_gs()
    ws = gs_client.open_by_key(sheet_id).worksheet(ws_name)
    df = pd.DataFrame(ws.get_all_records())
    return df, ws

def df_to_sheet(df: pd.DataFrame, ws):
    ws.update([df.columns.tolist()] + df.values.tolist())

# Process batches with real-time updates
def process_batches(df: pd.DataFrame, batch_size: int, delay: float, ws=None):
    total = math.ceil(len(df) / batch_size)
    progress = st.progress(0)
    status = st.empty()
    col_idx = df.columns.get_loc('Optimized Title') + 1 if 'Optimized Title' in df.columns else None
    if 'Optimized Title' not in df.columns:
        df['Optimized Title'] = ''

    for i, start in enumerate(range(0, len(df), batch_size), start=1):
        st.write(f"▶️ Starting batch {i}/{total}...")
        subset = df.iloc[start:start+batch_size]
        batch = []
        for _, row in subset.iterrows():
            batch.append({
                'Original Title': row['Original Title'],
                'Product Description': row.get('Product Description', ''),
                'Product Type': row['Product Type'],
                'Key Attributes': row['Key Attributes'],
                'Use Case': row['Use Case'],
                'PDP Content': fetch_pdp_text(row['PDP URL'])
            })

        status.text(f"Processing batch {i}/{total}...")
        with st.spinner(f"Calling OpenAI for batch {i}/{total}..."):
            resp = call_openai_with_retries(build_batch_prompt(batch))
        st.code(resp.choices[0].message.content, language='markdown')

        # parse out the titles
        titles = []
        lines = [line.strip() for line in resp.choices[0].message.content.splitlines() if line.strip()]

        for line in lines:
            if re.match(r'^\d+\.', line):
                parts = line.split('.', 1)
                if len(parts) > 1 and parts[1].strip():
                    titles.append(parts[1].strip())

        # fallback for single unnumbered line response
        if len(lines) == 1 and len(titles) == 0 and len(batch) == 1:
            titles = [lines[0]]

        # pad out missing titles
        titles += ['N/A'] * (len(batch) - len(titles))

        # write back to both DataFrame and sheet
        for idx, title in zip(subset.index, titles):
            st.write(f"🛠 Writing to sheet: row={idx+2}, col={col_idx}, title='{title}'")
            df.at[idx, 'Optimized Title'] = title
            st.toast(f"✅ Generated title for row {idx+2}: {title}")
            if ws and col_idx:
                try:
                    ws.update_cell(idx + 2, col_idx, title)
                except Exception as e:
                    st.warning(f"⚠️ Failed to write to sheet at row {idx+2}: {e}")

# Main function
def main():
    st.title("🔍 Product Title Optimizer")
    init_openai()
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Sheet Source", ["Existing Sheet", "Create New Sheet"])

    if mode == "Create New Sheet":
        if st.sidebar.button("Create Sheet from Template"):
            new_id = create_new_sheet()
            st.session_state['sheet_id'] = new_id
            st.success(f"Created new sheet ID: {new_id}")

    # Inputs for existing or new sheet
    sheet_id = st.sidebar.text_input("Google Sheet ID", st.session_state.get('sheet_id', ''))
    ws_name = st.sidebar.text_input("Worksheet Name", "Sheet1")
    batch_size = st.sidebar.slider("Batch size", 1, 10, 5)
    delay = st.sidebar.number_input("Delay (s)", 0.0, 30.0, 2.0)

    # Run optimization
    if st.sidebar.button("Start Optimization"):
        if not sheet_id:
            st.error("Enter a valid Google Sheet ID.")
            return
        df, ws = sheet_to_df(sheet_id, ws_name)
        process_batches(df, batch_size, delay, ws)
        st.success("Optimization complete!")

if __name__ == '__main__':
    main()
