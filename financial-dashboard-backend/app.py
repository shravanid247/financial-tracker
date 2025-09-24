"""
Full-Stack Fintech Dashboard
A complete multi-page Dash application with CRUD functionality.
Features: Dashboard, Transactions Management, and Portfolio Management.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging

# Try to import requests, fall back to mock data if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available. Using mock data.")

# Try to import financial logic functions, fall back to inline implementations if not available
try:
    from financial_logic import calculate_net_worth_timeline, calculate_asset_allocation, run_monte_carlo_simulation
    FINANCIAL_LOGIC_AVAILABLE = True
except ImportError:
    FINANCIAL_LOGIC_AVAILABLE = False
    logging.warning("Financial logic module not available. Using inline implementations.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI backend URL
API_BASE_URL = "http://127.0.0.1:8000"

# Define modern fintech color scheme
COLORS = {
    'background': '#0a0e1a',
    'card_background': '#1a1e2e', 
    'text': '#ffffff',
    'accent': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#4ecdc4',
    'warning': '#ffe66d',
    'muted': '#6c7293',
    'border': '#2d3142'
}

# Mock data for when backend is not available
MOCK_TRANSACTIONS = [
    {'id': 1, 'date': '2024-01-01T00:00:00', 'amount': 5000.0, 'category': 'Salary', 'description': 'Monthly salary'},
    {'id': 2, 'date': '2024-01-05T00:00:00', 'amount': -1200.0, 'category': 'Rent', 'description': 'Monthly rent'},
    {'id': 3, 'date': '2024-01-10T00:00:00', 'amount': -300.0, 'category': 'Food', 'description': 'Groceries'},
    {'id': 4, 'date': '2024-01-15T00:00:00', 'amount': 2000.0, 'category': 'Investment', 'description': 'Stock dividend'},
    {'id': 5, 'date': '2024-01-20T00:00:00', 'amount': -500.0, 'category': 'Utilities', 'description': 'Electric bill'},
]

MOCK_PORTFOLIO = {
    'total_value': 50000.0,
    'holdings': [
        {'ticker': 'AAPL', 'shares': 50.0, 'price_per_share': 180.50, 'total_value': 9025.0},
        {'ticker': 'GOOGL', 'shares': 10.0, 'price_per_share': 2800.00, 'total_value': 28000.0},
        {'ticker': 'MSFT', 'shares': 25.0, 'price_per_share': 380.75, 'total_value': 9518.75},
        {'ticker': 'TSLA', 'shares': 15.0, 'price_per_share': 250.30, 'total_value': 3754.50},
    ]
}

# Inline financial logic functions (fallback implementations)
def calculate_net_worth_timeline_inline(transactions):
    """Fallback implementation of net worth timeline calculation."""
    if not transactions:
        return pd.DataFrame(columns=['date', 'cumulative_net_worth'])
    
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['cumulative_net_worth'] = df['amount'].cumsum()
    return df[['date', 'cumulative_net_worth']].copy()

def run_monte_carlo_simulation_inline(initial_investment=100000, time_period=20, 
                                    expected_return=0.08, volatility=0.15, num_simulations=500):
    """Fallback implementation of Monte Carlo simulation."""
    np.random.seed(42)
    time_steps = time_period * 12
    dt = 1/12
    simulation_results = np.zeros((time_steps + 1, num_simulations))
    simulation_results[0, :] = initial_investment
    
    for t in range(1, time_steps + 1):
        random_shocks = np.random.normal(0, 1, num_simulations)
        drift = (expected_return - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt) * random_shocks
        simulation_results[t, :] = simulation_results[t-1, :] * np.exp(drift + diffusion)
    
    time_index = np.linspace(0, time_period, time_steps + 1)
    column_names = [f'Simulation_{i+1}' for i in range(num_simulations)]
    return pd.DataFrame(simulation_results, columns=column_names, index=time_index)

# Initialize Dash app
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Custom CSS styles (keeping the same as original)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Fintech Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                background-color: #0a0e1a;
                font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #ffffff;
            }
            
            .navbar {
                background: linear-gradient(135deg, rgba(26, 30, 46, 0.95), rgba(26, 30, 46, 1));
                backdrop-filter: blur(10px);
                padding: 1rem 2rem;
                border-bottom: 2px solid #2d3142;
                box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
            }
            
            .navbar-brand {
                color: #00d4ff;
                font-size: 1.8rem;
                font-weight: 700;
                text-decoration: none;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            }
            
            .navbar-nav {
                display: flex;
                list-style: none;
                gap: 2rem;
                margin-top: 1rem;
            }
            
            .nav-link {
                color: #6c7293;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            
            .nav-link:hover, .nav-link.active {
                color: #00d4ff;
                background-color: rgba(0, 212, 255, 0.1);
                transform: translateY(-2px);
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            .page-header {
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .page-title {
                color: #ffffff;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .page-subtitle {
                color: #6c7293;
                font-size: 1.1rem;
                font-weight: 300;
            }
            
            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-gap: 2rem;
                margin-bottom: 2rem;
            }
            
            .chart-card, .form-card {
                background: linear-gradient(135deg, rgba(26, 30, 46, 0.8), rgba(26, 30, 46, 1));
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
                border: 1px solid #2d3142;
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .chart-card:hover, .form-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
            }
            
            .full-width {
                grid-column: 1 / -1;
            }
            
            .card-title {
                color: #ffffff;
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            
            .form-label {
                color: #ffffff;
                font-weight: 500;
                margin-bottom: 0.5rem;
                display: block;
            }
            
            .form-input {
                width: 100%;
                padding: 0.75rem 1rem;
                background-color: rgba(45, 49, 66, 0.5);
                border: 2px solid #2d3142;
                border-radius: 8px;
                color: #ffffff;
                font-family: 'Inter', sans-serif;
                transition: all 0.3s ease;
            }
            
            .form-input:focus {
                outline: none;
                border-color: #00d4ff;
                box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
            }
            
            .btn {
                padding: 0.75rem 2rem;
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
                background: linear-gradient(135deg, #0099cc, #00d4ff);
            }
            
            .btn-secondary {
                background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
            }
            
            .btn-secondary:hover {
                background: linear-gradient(135deg, #ee5a5a, #ff6b6b);
                box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
            }
            
            .data-table {
                margin-top: 1rem;
            }
            
            .alert {
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                font-weight: 500;
            }
            
            .alert-success {
                background-color: rgba(78, 205, 196, 0.1);
                border: 1px solid #4ecdc4;
                color: #4ecdc4;
            }
            
            .alert-error {
                background-color: rgba(255, 107, 107, 0.1);
                border: 1px solid #ff6b6b;
                color: #ff6b6b;
            }
            
            @media (max-width: 768px) {
                .dashboard-grid {
                    grid-template-columns: 1fr;
                }
                
                .navbar {
                    padding: 1rem;
                }
                
                .navbar-nav {
                    flex-direction: column;
                    gap: 0.5rem;
                }
                
                .container {
                    padding: 1rem;
                }
                
                .page-title {
                    font-size: 2rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def get_api_data(endpoint):
    """Fetch data from FastAPI backend with error handling and fallback."""
    if not REQUESTS_AVAILABLE:
        logger.warning("Requests not available, using mock data")
        return get_mock_data(endpoint)
    
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {endpoint}: {str(e)}. Using mock data.")
        return get_mock_data(endpoint)

def post_api_data(endpoint, data):
    """Post data to FastAPI backend with error handling and fallback."""
    if not REQUESTS_AVAILABLE:
        logger.warning("Requests not available, simulating successful post")
        return {"success": True, "message": "Mock post successful"}
    
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API post failed for {endpoint}: {str(e)}")
        return {"success": False, "message": str(e)}

def get_mock_data(endpoint):
    """Return mock data based on endpoint."""
    if endpoint == "/api/transactions":
        return MOCK_TRANSACTIONS
    elif endpoint == "/api/portfolio/value":
        return MOCK_PORTFOLIO
    elif endpoint == "/api/portfolio":
        return [{'id': i+1, 'ticker': h['ticker'], 'shares': h['shares']} 
                for i, h in enumerate(MOCK_PORTFOLIO['holdings'])]
    return None

# Navigation component
def create_navbar():
    """Create navigation bar component."""
    return html.Div([
        html.Div([
            dcc.Link("💰 Fintech Dashboard", href="/", className="navbar-brand"),
            html.Nav([
                dcc.Link([html.I(className="fas fa-chart-line"), " Dashboard"], 
                        href="/", className="nav-link"),
                dcc.Link([html.I(className="fas fa-exchange-alt"), " Transactions"], 
                        href="/transactions", className="nav-link"),
                dcc.Link([html.I(className="fas fa-briefcase"), " Portfolio"], 
                        href="/portfolio", className="nav-link"),
            ], className="navbar-nav")
        ], style={"display": "flex", "justify-content": "space-between", "align-items": "center"})
    ], className="navbar")

# Dashboard page layout
def dashboard_layout():
    """Create the main dashboard page layout."""
    return html.Div([
        create_navbar(),
        html.Div([
            html.Div([
                html.H1("Financial Dashboard", className="page-title"),
                html.P("Advanced Analytics & Portfolio Management", className="page-subtitle")
            ], className="page-header"),
            
            html.Div([
                # Net Worth Timeline Chart
                html.Div([
                    html.H3("Net Worth Timeline", className="card-title"),
                    dcc.Graph(id='net-worth-graph', config={'displayModeBar': False})
                ], className="chart-card"),
                
                # Asset Allocation Chart
                html.Div([
                    html.H3("Asset Allocation", className="card-title"),
                    dcc.Graph(id='asset-allocation-graph', config={'displayModeBar': False})
                ], className="chart-card"),
                
                # Monte Carlo Simulation Chart
                html.Div([
                    html.H3("Monte Carlo Portfolio Projection (20 Years)", className="card-title"),
                    dcc.Graph(id='monte-carlo-graph', config={'displayModeBar': False})
                ], className="chart-card full-width"),
            ], className="dashboard-grid"),
            
            # Hidden div to trigger callbacks
            html.Div(id="dashboard-trigger", style={"display": "none"})
        ], className="container")
    ])

# Transactions page layout
def transactions_layout():
    """Create the transactions management page layout."""
    return html.Div([
        create_navbar(),
        html.Div([
            html.Div([
                html.H1("Transaction Management", className="page-title"),
                html.P("View and manage your financial transactions", className="page-subtitle")
            ], className="page-header"),
            
            html.Div([
                # Add Transaction Form
                html.Div([
                    html.H3("Add New Transaction", className="card-title"),
                    html.Div(id="transaction-alert"),
                    
                    html.Div([
                        html.Label("Amount ($)", className="form-label"),
                        dcc.Input(
                            id="transaction-amount",
                            type="number",
                            placeholder="Enter amount (positive for income, negative for expense)",
                            className="form-input",
                            step=0.01
                        )
                    ], className="form-group"),
                    
                    html.Div([
                        html.Label("Category", className="form-label"),
                        dcc.Input(
                            id="transaction-category",
                            type="text",
                            placeholder="e.g., Salary, Food, Rent, Investment",
                            className="form-input"
                        )
                    ], className="form-group"),
                    
                    html.Div([
                        html.Label("Description", className="form-label"),
                        dcc.Input(
                            id="transaction-description",
                            type="text",
                            placeholder="Brief description of the transaction",
                            className="form-input"
                        )
                    ], className="form-group"),
                    
                    html.Button("Add Transaction", id="add-transaction-btn", className="btn")
                ], className="form-card"),
                
                # Transactions Table
                html.Div([
                    html.H3("Transaction History", className="card-title"),
                    html.Div(id="transactions-table-container")
                ], className="chart-card full-width")
            ], className="dashboard-grid"),
            
            # Hidden divs for callbacks
            html.Div(id="transactions-trigger", style={"display": "none"})
        ], className="container")
    ])

# Portfolio page layout
def portfolio_layout():
    """Create the portfolio management page layout."""
    return html.Div([
        create_navbar(),
        html.Div([
            html.Div([
                html.H1("Portfolio Management", className="page-title"),
                html.P("Manage your investment holdings", className="page-subtitle")
            ], className="page-header"),
            
            html.Div([
                # Add Holding Form
                html.Div([
                    html.H3("Add/Update Holding", className="card-title"),
                    html.Div(id="portfolio-alert"),
                    
                    html.Div([
                        html.Label("Stock Ticker", className="form-label"),
                        dcc.Input(
                            id="portfolio-ticker",
                            type="text",
                            placeholder="e.g., AAPL, GOOGL, MSFT",
                            className="form-input"
                        )
                    ], className="form-group"),
                    
                    html.Div([
                        html.Label("Number of Shares", className="form-label"),
                        dcc.Input(
                            id="portfolio-shares",
                            type="number",
                            placeholder="Enter number of shares",
                            className="form-input",
                            min=0,
                            step=0.01
                        )
                    ], className="form-group"),
                    
                    html.Button("Add/Update Holding", id="add-holding-btn", className="btn")
                ], className="form-card"),
                
                # Portfolio Value Display
                html.Div([
                    html.H3("Portfolio Summary", className="card-title"),
                    html.Div(id="portfolio-summary")
                ], className="chart-card"),
                
                # Portfolio Holdings Table
                html.Div([
                    html.H3("Current Holdings", className="card-title"),
                    html.Div(id="portfolio-table-container")
                ], className="chart-card full-width")
            ], className="dashboard-grid"),
            
            # Hidden divs for callbacks
            html.Div(id="portfolio-trigger", style={"display": "none"})
        ], className="container")
    ])

# Main app layout with routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Routing callback
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    """Handle page routing based on URL pathname."""
    if pathname == '/transactions':
        return transactions_layout()
    elif pathname == '/portfolio':
        return portfolio_layout()
    else:  # Default to dashboard
        return dashboard_layout()

# Dashboard callbacks
@app.callback(
    Output('net-worth-graph', 'figure'),
    Input('dashboard-trigger', 'children')
)
def update_net_worth_timeline(_):
    """Update net worth timeline chart."""
    try:
        transactions_data = get_api_data("/api/transactions")
        
        if not transactions_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No transaction data available<br>Add some transactions to see your net worth timeline",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(color=COLORS['muted'], size=16)
            )
        else:
            # Use appropriate function based on availability
            if FINANCIAL_LOGIC_AVAILABLE:
                timeline_df = calculate_net_worth_timeline(transactions_data)
            else:
                timeline_df = calculate_net_worth_timeline_inline(transactions_data)
            
            if timeline_df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No timeline data to display",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    xanchor='center', yanchor='middle', showarrow=False,
                    font=dict(color=COLORS['muted'], size=16)
                )
            else:
                fig = px.line(timeline_df, x='date', y='cumulative_net_worth', template="plotly_dark")
                fig.update_traces(
                    line=dict(color=COLORS['accent'], width=3),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Net Worth:</b> $%{y:,.2f}<extra></extra>"
                )
                fig.add_scatter(
                    x=timeline_df['date'], y=timeline_df['cumulative_net_worth'],
                    fill='tonexty', mode='none', fillcolor='rgba(0, 212, 255, 0.1)', showlegend=False
                )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'], family="Inter"), height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, showline=True, linecolor=COLORS['border'],
                      title="Date", title_font=dict(color=COLORS['muted'])),
            yaxis=dict(showgrid=True, gridcolor='rgba(45, 49, 66, 0.4)',
                      showline=True, linecolor=COLORS['border'],
                      title="Net Worth ($)", title_font=dict(color=COLORS['muted']),
                      tickformat="$,.0f"),
            hovermode='x unified'
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating net worth timeline: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data: {str(e)}", xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle', showarrow=False,
            font=dict(color=COLORS['secondary'], size=14)
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

@app.callback(
    Output('asset-allocation-graph', 'figure'),
    Input('dashboard-trigger', 'children')
)
def update_asset_allocation(_):
    """Update asset allocation donut chart."""
    try:
        portfolio_data = get_api_data("/api/portfolio/value")
        
        if not portfolio_data or not portfolio_data.get('holdings'):
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available<br>Add some holdings to see allocation",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(color=COLORS['muted'], size=16)
            )
        else:
            holdings = portfolio_data['holdings']
            total_value = sum(holding['total_value'] for holding in holdings)
            tickers = [holding['ticker'] for holding in holdings]
            values = [holding['total_value'] for holding in holdings]
            percentages = [(value/total_value)*100 for value in values]
            
            fig = go.Figure(data=[go.Pie(
                labels=tickers, values=percentages, hole=0.6,
                hovertemplate="<b>%{label}</b><br>Allocation: %{value:.1f}%<br>Value: $%{customdata:,.2f}<extra></extra>",
                customdata=values, textinfo='label+percent', textposition='outside',
                textfont=dict(color=COLORS['text'], size=12),
                marker=dict(
                    colors=[COLORS['accent'], COLORS['success'], COLORS['warning'], 
                           COLORS['secondary'], '#9b59b6', '#e67e22', '#1abc9c', '#34495e'],
                    line=dict(color=COLORS['background'], width=2)
                )
            )])
            
            fig.add_annotation(
                text=f"Total<br><b>${total_value:,.0f}</b>", x=0.5, y=0.5,
                font=dict(size=18, color=COLORS['text']), showarrow=False
            )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'], family="Inter"), height=400,
            margin=dict(l=20, r=20, t=20, b=20), showlegend=False
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating asset allocation chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading portfolio data: {str(e)}", xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle', showarrow=False,
            font=dict(color=COLORS['secondary'], size=14)
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

@app.callback(
    Output('monte-carlo-graph', 'figure'),
    Input('dashboard-trigger', 'children')
)
def update_monte_carlo_simulation(_):
    """Update Monte Carlo simulation chart."""
    try:
        # Use appropriate function based on availability
        if FINANCIAL_LOGIC_AVAILABLE:
            simulation_df = run_monte_carlo_simulation(
                initial_investment=100000, time_period=20,
                expected_return=0.08, volatility=0.15, num_simulations=500
            )
        else:
            simulation_df = run_monte_carlo_simulation_inline(
                initial_investment=100000, time_period=20,
                expected_return=0.08, volatility=0.15, num_simulations=500
            )
        
        fig = go.Figure()
        
        # Add sample simulation paths
        sample_simulations = simulation_df.iloc[:, ::20]
        for col in sample_simulations.columns:
            fig.add_trace(go.Scatter(
                x=simulation_df.index, y=simulation_df[col], mode='lines',
                line=dict(color='rgba(108, 114, 147, 0.3)', width=0.5),
                hoverinfo='skip', showlegend=False
            ))
        
        # Calculate percentiles
        percentiles = simulation_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=simulation_df.index, y=percentiles[0.95], mode='lines',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=simulation_df.index, y=percentiles[0.05], mode='lines',
            line=dict(color='rgba(0,0,0,0)'), fill='tonexty',
            fillcolor='rgba(0, 212, 255, 0.1)', name='90% Confidence',
            hovertemplate="<b>Year:</b> %{x}<br><b>Range:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        # Add median and mean lines
        fig.add_trace(go.Scatter(
            x=simulation_df.index, y=percentiles[0.5], mode='lines',
            line=dict(color=COLORS['accent'], width=3), name='Median Projection',
            hovertemplate="<b>Year:</b> %{x}<br><b>Median Value:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        mean_values = simulation_df.mean(axis=1)
        fig.add_trace(go.Scatter(
            x=simulation_df.index, y=mean_values, mode='lines',
            line=dict(color=COLORS['success'], width=2, dash='dash'), name='Mean Projection',
            hovertemplate="<b>Year:</b> %{x}<br><b>Mean Value:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'], family="Inter"), height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, showline=True, linecolor=COLORS['border'],
                      title="Years", title_font=dict(color=COLORS['muted'])),
            yaxis=dict(showgrid=True, gridcolor='rgba(45, 49, 66, 0.4)',
                      showline=True, linecolor=COLORS['border'],
                      title="Portfolio Value ($)", title_font=dict(color=COLORS['muted']),
                      tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                       xanchor="right", x=1, bgcolor='rgba(0,0,0,0)',
                       font=dict(color=COLORS['text'])),
            hovermode='x unified'
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Monte Carlo simulation: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating simulation: {str(e)}", xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle', showarrow=False,
            font=dict(color=COLORS['secondary'], size=14)
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        return fig

# Transactions page callbacks
@app.callback(
    Output('transactions-table-container', 'children'),
    [Input('transactions-trigger', 'children'),
     Input('add-transaction-btn', 'n_clicks')]
)
def update_transactions_table(_, n_clicks):
    """Update transactions table."""
    try:
        transactions = get_api_data("/api/transactions")
        
        if not transactions:
            return html.Div("No transactions found. Add some transactions to get started!", 
                          style={'color': COLORS['muted'], 'text-align': 'center', 'padding': '2rem'})
        
        # Format data for table
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
        df['amount'] = df['amount'].apply(lambda x: f"${x:,.2f}")
        
        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[
                {'name': 'Date', 'id': 'date'},
                {'name': 'Amount', 'id': 'amount'},
                {'name': 'Category', 'id': 'category'},
                {'name': 'Description', 'id': 'description'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': 'rgba(45, 49, 66, 0.5)',
                'color': '#ffffff',
                'border': '1px solid #2d3142',
                'textAlign': 'left',
                'padding': '12px',
                'fontFamily': 'Inter'
            },
            style_header={
                'backgroundColor': 'rgba(0, 212, 255, 0.1)',
                'color': '#00d4ff',
                'fontWeight': 'bold',
                'border': '1px solid #00d4ff'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(26, 30, 46, 0.5)'
                }
            ]
        )
    except Exception as e:
        return html.Div(f"Error loading transactions: {str(e)}", 
                       style={'color': COLORS['secondary'], 'text-align': 'center', 'padding': '2rem'})

@app.callback(
    [Output('transaction-alert', 'children'),
     Output('transactions-trigger', 'children'),
     Output('transaction-amount', 'value'),
     Output('transaction-category', 'value'),
     Output('transaction-description', 'value')],  
    [Input('add-transaction-btn', 'n_clicks')],
    [State('transaction-amount', 'value'),
     State('transaction-category', 'value'),
     State('transaction-description', 'value')]
)
def add_transaction(n_clicks, amount, category, description):
    """Add a new transaction to the database."""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Validation
    if not amount or amount == 0:
        alert = html.Div("Please enter a valid amount (cannot be zero)", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if not category or not category.strip():
        alert = html.Div("Please enter a category", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if not description or not description.strip():
        alert = html.Div("Please enter a description", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Prepare transaction data
    transaction_data = {
        'amount': float(amount),
        'category': category.strip(),
        'description': description.strip(),
        'date': datetime.now().isoformat()
    }
    
    # Post to API
    result = post_api_data("/api/transactions", transaction_data)
    
    if result and result.get('success', True):
        alert = html.Div("Transaction added successfully!", className="alert alert-success")
        # Clear form and trigger table refresh
        return alert, datetime.now().isoformat(), None, '', ''
    else:
        error_msg = result.get('message', 'Unknown error') if result else 'Failed to connect to backend'
        alert = html.Div(f"Error adding transaction: {error_msg}", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Portfolio page callbacks
@app.callback(
    Output('portfolio-table-container', 'children'),
    [Input('portfolio-trigger', 'children'),
     Input('add-holding-btn', 'n_clicks')]
)
def update_portfolio_table(_, n_clicks):
    """Update portfolio table."""
    try:
        holdings = get_api_data("/api/portfolio")
        
        if not holdings:
            return html.Div("No portfolio holdings found. Add some holdings to get started!", 
                          style={'color': COLORS['muted'], 'text-align': 'center', 'padding': '2rem'})
        
        # Format data for table
        df = pd.DataFrame(holdings)
        
        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[
                {'name': 'Ticker', 'id': 'ticker'},
                {'name': 'Shares', 'id': 'shares', 'type': 'numeric', 'format': {'specifier': ',.2f'}}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': 'rgba(45, 49, 66, 0.5)',
                'color': '#ffffff',
                'border': '1px solid #2d3142',
                'textAlign': 'left',
                'padding': '12px',
                'fontFamily': 'Inter'
            },
            style_header={
                'backgroundColor': 'rgba(0, 212, 255, 0.1)',
                'color': '#00d4ff',
                'fontWeight': 'bold',
                'border': '1px solid #00d4ff'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(26, 30, 46, 0.5)'
                }
            ]
        )
    except Exception as e:
        return html.Div(f"Error loading portfolio: {str(e)}", 
                       style={'color': COLORS['secondary'], 'text-align': 'center', 'padding': '2rem'})

@app.callback(
    Output('portfolio-summary', 'children'),
    [Input('portfolio-trigger', 'children'),
     Input('add-holding-btn', 'n_clicks')]
)
def update_portfolio_summary(_, n_clicks):
    """Update portfolio summary display."""
    try:
        portfolio_data = get_api_data("/api/portfolio/value")
        
        if not portfolio_data:
            return html.Div("No portfolio data available", 
                          style={'color': COLORS['muted'], 'text-align': 'center', 'padding': '2rem'})
        
        total_value = portfolio_data.get('total_value', 0)
        num_holdings = len(portfolio_data.get('holdings', []))
        
        return html.Div([
            html.H2(f"${total_value:,.2f}", style={'color': COLORS['accent'], 'text-align': 'center', 'margin': '0'}),
            html.P(f"Total Portfolio Value", style={'color': COLORS['text'], 'text-align': 'center', 'margin': '0.5rem 0 0 0'}),
            html.P(f"{num_holdings} Holdings", style={'color': COLORS['muted'], 'text-align': 'center', 'margin': '0.5rem 0 0 0'})
        ])
        
    except Exception as e:
        return html.Div(f"Error loading portfolio summary: {str(e)}", 
                       style={'color': COLORS['secondary'], 'text-align': 'center', 'padding': '2rem'})

@app.callback(
    [Output('portfolio-alert', 'children'),
     Output('portfolio-trigger', 'children'),
     Output('portfolio-ticker', 'value'),
     Output('portfolio-shares', 'value')],
    [Input('add-holding-btn', 'n_clicks')],
    [State('portfolio-ticker', 'value'),
     State('portfolio-shares', 'value')]
)
def add_holding(n_clicks, ticker, shares):
    """Add or update a portfolio holding."""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Validation
    if not ticker or not ticker.strip():
        alert = html.Div("Please enter a stock ticker", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update
    
    if not shares or shares <= 0:
        alert = html.Div("Please enter a valid number of shares (must be positive)", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update
    
    # Prepare holding data
    holding_data = {
        'ticker': ticker.strip().upper(),
        'shares': float(shares)
    }
    
    # Post to API
    result = post_api_data("/api/portfolio", holding_data)
    
    if result and result.get('success', True):
        alert = html.Div("Holding added/updated successfully!", className="alert alert-success")
        # Clear form and trigger table refresh
        return alert, datetime.now().isoformat(), '', None
    else:
        error_msg = result.get('message', 'Unknown error') if result else 'Failed to connect to backend'
        alert = html.Div(f"Error adding/updating holding: {error_msg}", className="alert alert-error")
        return alert, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)