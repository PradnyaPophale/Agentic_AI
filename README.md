📊 Test Execution Analysis Dashboard

A Streamlit-based AI-powered dashboard for analyzing test execution reports using AutoGen agents and Groq LLMs. This system automates validation, failure categorization, and actionable recommendations, enabling faster insights from test execution data.

🚀 Features

AI-powered Analysis Pipeline

Validation Agent: Validates CSV structure and generates summary metrics

Failure Analysis Agent: Categorizes failed test cases and assigns team accountability

Action Points Agent: Generates prioritized action recommendations and long-term improvements

Interactive Visualizations

Pass/Fail pie chart

Suite-wise bar chart

Failure categories with team accountability

Streamlit Dashboard

File upload support for CSV test execution reports

Real-time metrics and visual analytics

Expanders for category details, sample failures, and recommendations

Extensible & Modular

Add new agents or modify analysis rules easily

Compatible with Groq LLM API for intelligent reasoning

🛠️ Technologies Used

Languages: Python, SQL

AI & Agentic Technologies: AutoGen, Groq LLMs, Agentic AI

Data Processing & Visualization: Pandas, Plotly, Plotly Express

Web Framework: Streamlit

Automation & APIs: REST API integration via agents


Install dependencies

pip install -r requirements.txt
Set your Groq API key
You can enter the API key directly in the Streamlit sidebar or set it as an environment variable:

export GROQ_API_KEY="your_api_key"      # Linux/Mac
set GROQ_API_KEY="your_api_key"         # Windows

📂 Usage

Run the Streamlit app

streamlit run app.py


Upload CSV File

Ensure the CSV has these required columns:

TMSID
TC_NAME
STATUS
FAILURE_REASON
SUITE_NAME
COMMENTS


Generate Analysis

Click “Generate Analysis” to trigger the AI pipeline

View metrics, visualizations, failure categorizations, and action points

Explore Results

Expand categories to see sample failures and team accountability

Check immediate actions and long-term improvement suggestions

📝 Sample CSV Format
TMSID	TC_NAME	STATUS	FAILURE_REASON	SUITE_NAME	COMMENTS
TMS001	Login Test	PASS		Smoke	Success
TMS002	API Test	FAIL	500 Internal Server Error	Regression	Server down
TMS003	UI Test	PASS		Smoke	Success
⚡ How It Works

Validation Agent

Checks required columns

Computes summary metrics (total tests, pass/fail count, pass percentage, suite breakdown)

Failure Analysis Agent

Analyzes failed test cases

Categorizes failures (Server, UI, Database, DevOps, etc.)

Assigns responsible teams

Action Points Agent

Prioritizes high-impact failure categories

Generates immediate actions and long-term improvement recommendations

Visualization & Dashboard

Interactive charts using Plotly

Metrics cards and expandable sections for detailed insights

💡 Tips & Troubleshooting

Ensure Groq API key is valid and has available credits.

Check CSV file format and required columns before analysis.

Reduce CSV size if experiencing timeouts with very large datasets.

Use the debug expander in the app to view raw agent responses for troubleshooting.

📦 Output

Summary metrics (total, passed, failed, pass percentage)

Suite-wise visualizations

Failure categorization with team accountability

Priority categories and recommended actions

📌 Future Enhancements

Export analysis results as downloadable JSON/Excel

Support for multiple LLM backends (OpenAI, Hugging Face, etc.)

Integration with Jira/Teams for automated ticketing on failure categories

Historical trend analysis across multiple test runs

🛠️ Requirements

Python >= 3.9

Streamlit

Pandas

Plotly

AutoGen SDK (Groq LLM integration)