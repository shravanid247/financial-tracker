# Personal Finance Dashboard

A comprehensive command-line application for tracking personal finances, including transaction categorization, net worth monitoring, investment portfolio tracking, and financial visualizations.

## Features

- **Transaction Management**: Load and categorize bank/credit card transactions from CSV files
- **Monthly Analysis**: Calculate monthly income, expenses, and net income
- **Net Worth Tracking**: Monitor net worth changes over time
- **Investment Portfolio**: Track investment holdings and performance
- **Financial Reports**: Generate comprehensive summary reports
- **Data Visualization**: Create charts and graphs for key financial metrics

## Installation

1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your data files in the `data/` directory:
   - `transactions.csv`: Your bank/credit card transactions
   - `net_worth.json`: Your net worth history
   - `investments.json`: Your investment portfolio

2. Run the dashboard:
   ```bash
   python personal_finance_dashboard.py
   ```

## Data File Formats

### transactions.csv
```csv
date,description,amount,account
2024-01-15,Salary Deposit,5000.00,Checking
2024-01-16,Grocery Store,125.50,Checking
```

### net_worth.json
```json
[
  {
    "date": "2024-01-01",
    "net_worth": 25000.00
  }
]
```

### investments.json
```json
[
  {
    "symbol": "AAPL",
    "shares": 10.0,
    "purchase_price": 140.00
  }
]
```

## Features in Detail

### Transaction Categorization
The dashboard automatically categorizes transactions based on keywords in the description:
- Income: salary, bonus, dividend, interest, refund
- Housing: rent, mortgage, utilities, insurance
- Food: groceries, restaurant, dining, food
- Transportation: gas, car, uber, lyft, parking
- Healthcare: doctor, pharmacy, medical, health
- Entertainment: movie, netflix, spotify, game
- Shopping: amazon, store, clothing, electronics
- Education: tuition, book, course, education
- Savings: savings, investment, retirement

### Financial Metrics
- Monthly income and expense tracking
- Net income calculations
- Expense category breakdowns
- Investment portfolio performance
- Net worth trend analysis

### Visualizations
- Monthly income vs expenses bar chart
- Expense category pie chart
- Net worth over time line graph
- Investment holdings bar chart

## Sample Data

The project includes sample data files to demonstrate functionality. You can replace these with your own financial data.

## Requirements

- Python 3.7+
- matplotlib
- pandas
- requests

## Notes

- Investment prices are currently mocked for demonstration purposes
- In a production environment, you would integrate with real financial APIs
- The dashboard saves visualizations as PNG files in the data directory