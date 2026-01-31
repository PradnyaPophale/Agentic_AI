import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from autogen import AssistantAgent, UserProxyAgent
import json
import os
from typing import Dict, List, Any
import re

# Page configuration
st.set_page_config(
    page_title="Test Execution Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-output {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}


class TestAnalysisSystem:
    def __init__(self, api_key: str):
        """Initialize the AutoGen agents with Groq LLM configuration"""

        # LLM Configuration for Groq
        self.llm_config = {
            "config_list": [
                {
                    "model": "llama-3.3-70b-versatile",  # Updated to newer model
                    "api_key": api_key,
                    "api_type": "groq",
                    "base_url": "https://api.groq.com/openai/v1"
                }
            ],
            "temperature": 0.3,
            "timeout": 120,
        }

        # Agent 1: Validation and Summary Agent
        self.validation_agent = AssistantAgent(
            name="ValidationSummaryAgent",
            system_message="""You are a Test Data Validation and Summary Agent. Your responsibilities:

            1. Validate CSV columns: Check if TMSID, TC_NAME, STATUS, FAILURE_REASON, SUITE_NAME, COMMENTS exist
            2. If validation passes, provide detailed summary:
               - Total test cases
               - Total passed test cases
               - Total failed test cases
               - Pass percentage
               - Suite-wise breakdown

            Return ONLY a valid JSON object with this structure:
            {
                "validation_passed": true/false,
                "missing_columns": [],
                "total_tc": number,
                "tc_passed": number,
                "tc_failed": number,
                "pass_percentage": number,
                "suite_breakdown": {}
            }

            Be precise and analytical.""",
            llm_config=self.llm_config,
        )

        # Agent 2: Failure Analysis and Categorization Agent
        self.failure_agent = AssistantAgent(
            name="FailureCategorizationAgent",
            system_message="""You are a Failure Analysis and Categorization Agent. Your responsibilities:

            1. Analyze all FAILED test cases and their FAILURE_REASON
            2. Group similar failure reasons into logical categories
            3. Assign team accountability for each category

             Common categories and teams:
            - Server errors (5xx, timeout, connection): OSS Team
            - Authentication/Authorization (401, 403, token): Security Team
            - Validation errors (400, invalid data,event validation failed): Development Team
            - UI/Frontend issues (element not found, rendering): Frontend Team
            - Database errors (SQL, connection, query): Database Team
            - Environment issues (config, setup): DevOps Team
            - Seagull related issues: Seagull Team
            - Others(400,402,422, invalid data): Test Design Team

            Return ONLY a valid JSON object:
            {
                "failure_categories": [
                    {
                        "category": "category_name",
                        "count": number,
                        "team_accountability": "team_name",
                        "sample_failures": ["reason1", "reason2"]
                    }
                ]
            }""",
            llm_config=self.llm_config,
        )

        # Agent 3: Action Points and Recommendations Agent
        self.action_agent = AssistantAgent(
            name="ActionPointsAgent",
            system_message="""You are an Action Points and Recommendations Agent. Your responsibilities:

            1. Analyze failure categories from previous agent
            2. Identify high-priority categories (highest failure counts)
            3. Generate actionable recommendations
            4. Prioritize action items

            Return ONLY a valid JSON object:
            {
                "priority_categories": [
                    {
                        "category": "name",
                        "priority": "High/Medium/Low",
                        "impact": "percentage or count",
                        "recommended_actions": ["action1", "action2"]
                    }
                ],
                "immediate_actions": ["action1", "action2"],
                "long_term_improvements": ["improvement1", "improvement2"]
            }""",
            llm_config=self.llm_config,
        )

        # User Proxy Agent
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )

    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from agent response"""
        # First, try to extract from markdown code blocks
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # ```json { } ```
            r'```\s*(\{.*?\})\s*```',  # ``` { } ```
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass

        # Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except:
            pass

        # Look for JSON object anywhere in the response
        json_match = re.search(r'\{[^{]*(?:\{[^{]*\}[^{]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # Return empty dict if all parsing fails
        return {}

    def analyze_test_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        results = {}

        # Step 1: Validation and Summary
        st.info("🔍 Agent 1: Validating data and generating summary...")

        required_columns = ['TMSID', 'TC_NAME', 'STATUS', 'FAILURE_REASON', 'SUITE_NAME', 'COMMENTS']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            results['validation'] = {
                "validation_passed": False,
                "missing_columns": missing_columns
            }
            return results

        # Normalize STATUS column to uppercase for consistent processing
        df['STATUS'] = df['STATUS'].str.upper().str.strip()

        # Prepare data summary for agent
        suite_breakdown = {}
        for suite in df['SUITE_NAME'].unique():
            suite_df = df[df['SUITE_NAME'] == suite]
            suite_breakdown[suite] = suite_df['STATUS'].value_counts().to_dict()

        data_summary = {
            "total_records": len(df),
            "status_counts": df['STATUS'].value_counts().to_dict(),
            "suite_breakdown": suite_breakdown
        }

        validation_prompt = f"""Analyze this test execution data and provide summary:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Provide validation result and comprehensive summary in JSON format."""

        self.user_proxy.initiate_chat(
            self.validation_agent,
            message=validation_prompt,
        )

        validation_response = self.validation_agent.last_message()["content"]
        results['validation'] = self.extract_json_from_response(validation_response)

        # Add actual calculations
        total_tc = len(df)
        tc_passed = len(df[df['STATUS'] == 'PASS'])
        tc_failed = len(df[df['STATUS'] == 'FAIL'])
        pass_percentage = round((tc_passed / total_tc * 100), 2) if total_tc > 0 else 0

        results['validation'].update({
            'total_tc': total_tc,
            'tc_passed': tc_passed,
            'tc_failed': tc_failed,
            'pass_percentage': pass_percentage,
            'validation_passed': True,
            'missing_columns': []
        })

        # Step 2: Failure Analysis
        st.info("🔎 Agent 2: Analyzing failures and categorizing...")

        failed_df = df[df['STATUS'] == 'FAIL'].copy()

        if len(failed_df) > 0:
            failure_reasons = failed_df['FAILURE_REASON'].dropna().tolist()

            failure_prompt = f"""Analyze these failure reasons and categorize them:

            Total Failed Cases: {len(failed_df)}

            Failure Reasons (sample of up to 50):
            {json.dumps(failure_reasons[:50], indent=2)}

            Categorize these failures, identify patterns, and assign team accountability in JSON format."""

            self.user_proxy.initiate_chat(
                self.failure_agent,
                message=failure_prompt,
            )

            failure_response = self.failure_agent.last_message()["content"]
            results['failures'] = self.extract_json_from_response(failure_response)
        else:
            results['failures'] = {"failure_categories": []}

        # Step 3: Action Points
        st.info("📋 Agent 3: Generating action points and recommendations...")

        action_prompt = f"""Based on the failure analysis below, generate prioritized action points:

        Summary:
        - Total Test Cases: {total_tc}
        - Failed Cases: {tc_failed}
        - Pass Rate: {pass_percentage}%

        Failure Categories:
        {json.dumps(results.get('failures', {}), indent=2)}

        Provide actionable recommendations in JSON format."""

        self.user_proxy.initiate_chat(
            self.action_agent,
            message=action_prompt,
        )

        action_response = self.action_agent.last_message()["content"]
        results['actions'] = self.extract_json_from_response(action_response)

        # Store raw responses for debugging
        results['raw_responses'] = {
            'validation': validation_response,
            'failures': failure_response if len(failed_df) > 0 else None,
            'actions': action_response
        }

        return results


def create_visualizations(df: pd.DataFrame, results: Dict):
    """Create interactive visualizations"""

    col1, col2 = st.columns(2)

    with col1:
        # Pass/Fail Pie Chart
        status_counts = df['STATUS'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker_colors=['#00cc96', '#ef553b'],
            textinfo='label+percent+value'
        )])
        fig_pie.update_layout(
            title="Test Execution Status Distribution",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Suite-wise breakdown
        suite_status = df.groupby(['SUITE_NAME', 'STATUS']).size().reset_index(name='count')
        fig_bar = px.bar(
            suite_status,
            x='SUITE_NAME',
            y='count',
            color='STATUS',
            title="Test Results by Suite",
            color_discrete_map={'PASS': '#00cc96', 'FAIL': '#ef553b'},
            barmode='group',
            height=400
        )
        fig_bar.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Failure Categories Chart (if available)
    if 'failures' in results and 'failure_categories' in results['failures']:
        categories = results['failures']['failure_categories']
        if categories:
            cat_df = pd.DataFrame(categories)

            fig_cat = px.bar(
                cat_df,
                x='category',
                y='count',
                color='team_accountability',
                title="Failure Categories and Team Accountability",
                height=400,
                text='count'
            )
            fig_cat.update_traces(textposition='outside')
            fig_cat.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)


def main():
    st.markdown('<p class="main-header">📊 Test Execution Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Powered by AutoGen + Groq LLM")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key (get it from https://console.groq.com)"
        )

        st.markdown("---")
        st.markdown("### 📋 Required CSV Columns")
        st.markdown("""
        - TMSID
        - TC_NAME
        - STATUS
        - FAILURE_REASON
        - SUITE_NAME
        - COMMENTS
        """)

        st.markdown("---")
        st.markdown("### 🤖 Agent Pipeline")
        st.markdown("""
        1. **Validation Agent**: Validates data structure
        2. **Failure Analysis Agent**: Categorizes failures
        3. **Action Points Agent**: Generates recommendations
        """)

    # File upload
    st.header("📤 Upload Test Execution Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your test execution CSV file"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ File uploaded successfully! ({len(df)} records)")

            with st.expander("📄 Preview Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # Generate Analysis Button
            if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):
                if not groq_api_key:
                    st.error("❌ Please enter your Groq API key in the sidebar!")
                else:
                    with st.spinner("🔄 Running AI-powered analysis..."):
                        try:
                            # Initialize analysis system
                            analysis_system = TestAnalysisSystem(groq_api_key)

                            # Run analysis
                            results = analysis_system.analyze_test_data(df)

                            st.session_state.results = results
                            st.session_state.analysis_complete = True

                            st.success("✅ Analysis completed successfully!")

                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")

                            # Provide helpful debugging information
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.write("**Error Type:**", type(e).__name__)
                                st.write("**Error Message:**", str(e))
                                st.code(str(e), language="text")

                                # Common solutions
                                st.markdown("### 💡 Possible Solutions:")
                                st.markdown("""
                                1. **Check your Groq API Key**: Ensure it's valid and has credits
                                2. **Verify CSV Format**: Make sure all required columns exist
                                3. **Check Internet Connection**: Groq API requires internet access
                                4. **Try Again**: Sometimes API calls timeout, just retry
                                5. **Reduce Data Size**: If CSV is very large, try with fewer rows
                                """)

                            st.exception(e)

            # Display results if analysis is complete
            if st.session_state.analysis_complete and st.session_state.results:
                results = st.session_state.results

                st.markdown("---")
                st.header("📊 Analysis Results")

                # Validation Results
                if 'validation' in results:
                    validation = results['validation']

                    if not validation.get('validation_passed', False):
                        st.error("❌ Validation Failed!")
                        st.warning(f"Missing columns: {', '.join(validation.get('missing_columns', []))}")
                    else:
                        st.success("✅ Validation Passed!")

                        # Metrics
                        st.subheader("📈 Summary Metrics")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Test Cases", validation.get('total_tc', 0))
                        with col2:
                            st.metric("Passed", validation.get('tc_passed', 0), delta="✓")
                        with col3:
                            st.metric("Failed", validation.get('tc_failed', 0), delta="✗")
                        with col4:
                            st.metric("Pass Rate", f"{validation.get('pass_percentage', 0)}%")

                        # Visualizations
                        st.markdown("---")
                        st.subheader("📊 Visual Analytics")
                        create_visualizations(df, results)

                # Failure Analysis
                st.markdown("---")
                if 'failures' in results and results['failures'].get('failure_categories'):
                    st.subheader("🔍 Failure Analysis & Categorization")

                    categories = results['failures']['failure_categories']

                    for idx, cat in enumerate(categories, 1):
                        with st.expander(f"**Category {idx}: {cat.get('category', 'Unknown')}**", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Failure Count", cat.get('count', 0))
                            with col2:
                                st.metric("Team Accountability", cat.get('team_accountability', 'N/A'))

                            if 'sample_failures' in cat and cat['sample_failures']:
                                st.markdown("**Sample Failures:**")
                                for failure in cat['sample_failures'][:3]:
                                    st.markdown(f"- {failure}")

                # Action Points
                st.markdown("---")
                if 'actions' in results:
                    st.subheader("🎯 Action Points & Recommendations")

                    actions = results['actions']

                    # Debug: Show if actions is empty
                    if not actions or len(actions) == 0:
                        st.warning("⚠️ No action points were generated. This might be due to parsing issues.")

                        # Show debug info
                        with st.expander("🔍 Debug: View Raw Agent Response"):
                            if 'raw_responses' in results and 'actions' in results['raw_responses']:
                                st.code(results['raw_responses']['actions'], language='text')

                    # Priority Categories
                    if 'priority_categories' in actions:
                        st.markdown("#### 🔴 Priority Categories")
                        for cat in actions['priority_categories']:
                            priority_color = {
                                'High': '🔴',
                                'Medium': '🟡',
                                'Low': '🟢'
                            }.get(cat.get('priority', 'Medium'), '⚪')

                            st.markdown(
                                f"**{priority_color} {cat.get('category', 'Unknown')} ({cat.get('priority', 'Medium')} Priority)**")
                            st.markdown(f"- Impact: {cat.get('impact', 'N/A')}")

                            if 'recommended_actions' in cat:
                                st.markdown("- Recommended Actions:")
                                for action in cat['recommended_actions']:
                                    st.markdown(f"  - {action}")
                            st.markdown("---")

                    # Immediate Actions
                    if 'immediate_actions' in actions and actions['immediate_actions']:
                        st.markdown("#### ⚡ Immediate Actions Required")
                        for action in actions['immediate_actions']:
                            st.markdown(f"- {action}")

                    # Long-term Improvements
                    if 'long_term_improvements' in actions and actions['long_term_improvements']:
                        st.markdown("#### 🚀 Long-term Improvements")
                        for improvement in actions['long_term_improvements']:
                            st.markdown(f"- {improvement}")

                # # Download Results
                # st.markdown("---")
                # st.subheader("💾 Download Results")
                #
                # results_json = json.dumps(results, indent=2)
                # st.download_button(
                #     label="📥 Download Analysis (JSON)",
                #     data=results_json,
                #     file_name="test_analysis_results.json",
                #     mime="application/json"
                # )

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload a CSV file to begin analysis")

        # Sample data format
        st.markdown("---")
        st.subheader("📝 Sample CSV Format")
        sample_df = pd.DataFrame({
            'TMSID': ['TMS001', 'TMS002', 'TMS003'],
            'TC_NAME': ['Login Test', 'API Test', 'UI Test'],
            'STATUS': ['PASS', 'FAIL', 'PASS'],
            'FAILURE_REASON': ['', '500 Internal Server Error', ''],
            'SUITE_NAME': ['Smoke', 'Regression', 'Smoke'],
            'COMMENTS': ['Success', 'Server down', 'Success']
        })
        st.dataframe(sample_df, use_container_width=True)


if __name__ == "__main__":
    main()