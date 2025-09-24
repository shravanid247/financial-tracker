#!/usr/bin/env python3
"""
Personal Finance Dashboard
A command-line application for tracking personal finances including:
- Transaction categorization and monthly income/expenses
- Net worth tracking over time
- Investment portfolio monitoring
- Financial summary reports and visualizations
"""

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Transaction:
    """Represents a financial transaction"""
    date: str
    description: str
    amount: float
    category: str = "Uncategorized"
    account: str = ""


@dataclass
class Investment:
    """Represents an investment holding"""
    symbol: str
    shares: float
    purchase_price: float
    current_price: float = 0.0
    
    @property
    def current_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def gain_loss(self) -> float:
        return self.current_value - (self.shares * self.purchase_price)


class PersonalFinanceDashboard:
    """Main dashboard class for personal finance tracking"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.transactions: List[Transaction] = []
        self.investments: List[Investment] = []
        self.net_worth_history: List[Tuple[str, float]] = []
        self.categories = {
            'Income': ['salary', 'bonus', 'dividend', 'interest', 'refund'],
            'Housing': ['rent', 'mortgage', 'utilities', 'insurance'],
            'Food': ['groceries', 'restaurant', 'dining', 'food'],
            'Transportation': ['gas', 'car', 'uber', 'lyft', 'parking'],
            'Healthcare': ['doctor', 'pharmacy', 'medical', 'health'],
            'Entertainment': ['movie', 'netflix', 'spotify', 'game'],
            'Shopping': ['amazon', 'store', 'clothing', 'electronics'],
            'Education': ['tuition', 'book', 'course', 'education'],
            'Savings': ['savings', 'investment', 'retirement'],
            'Other': []
        }
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
    def load_transactions(self, filename: str) -> None:
        """Load transactions from CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Transaction file {filepath} not found!")
            return
            
        self.transactions = []
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    transaction = Transaction(
                        date=row['date'],
                        description=row['description'],
                        amount=float(row['amount']),
                        account=row.get('account', '')
                    )
                    self.transactions.append(transaction)
                except (ValueError, KeyError) as e:
                    print(f"Error parsing transaction: {row}, Error: {e}")
                    continue
        
        print(f"Loaded {len(self.transactions)} transactions")
        self._categorize_transactions()
    
    def _categorize_transactions(self) -> None:
        """Automatically categorize transactions based on description"""
        for transaction in self.transactions:
            description_lower = transaction.description.lower()
            
            # Find matching category
            for category, keywords in self.categories.items():
                if any(keyword in description_lower for keyword in keywords):
                    transaction.category = category
                    break
                else:
                    transaction.category = 'Other'
    
    def load_net_worth_history(self, filename: str) -> None:
        """Load net worth history from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Net worth file {filepath} not found!")
            return
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.net_worth_history = [(entry['date'], entry['net_worth']) for entry in data]
        
        print(f"Loaded {len(self.net_worth_history)} net worth entries")
    
    def load_investments(self, filename: str) -> None:
        """Load investment portfolio from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Investment file {filepath} not found!")
            return
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.investments = []
            for investment_data in data:
                investment = Investment(
                    symbol=investment_data['symbol'],
                    shares=investment_data['shares'],
                    purchase_price=investment_data['purchase_price']
                )
                self.investments.append(investment)
        
        print(f"Loaded {len(self.investments)} investments")
        self._update_investment_prices()
    
    def _update_investment_prices(self) -> None:
        """Fetch current prices for investments (mock implementation)"""
        # In a real implementation, you would use a financial API like Alpha Vantage, Yahoo Finance, etc.
        # For this example, we'll use mock data
        mock_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 200.0,
            'SPY': 400.0
        }
        
        for investment in self.investments:
            investment.current_price = mock_prices.get(investment.symbol, investment.purchase_price)
    
    def calculate_monthly_summary(self, year: int, month: int) -> Dict:
        """Calculate monthly income and expenses"""
        monthly_transactions = [
            t for t in self.transactions
            if datetime.strptime(t.date, '%Y-%m-%d').year == year
            and datetime.strptime(t.date, '%Y-%m-%d').month == month
        ]
        
        income = sum(t.amount for t in monthly_transactions if t.amount > 0)
        expenses = sum(abs(t.amount) for t in monthly_transactions if t.amount < 0)
        
        # Categorized expenses
        category_totals = defaultdict(float)
        for t in monthly_transactions:
            if t.amount < 0:
                category_totals[t.category] += abs(t.amount)
        
        return {
            'income': income,
            'expenses': expenses,
            'net_income': income - expenses,
            'category_breakdown': dict(category_totals),
            'transaction_count': len(monthly_transactions)
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_value = sum(inv.current_value for inv in self.investments)
        total_cost = sum(inv.shares * inv.purchase_price for inv in self.investments)
        total_gain_loss = sum(inv.gain_loss for inv in self.investments)
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain_loss': total_gain_loss,
            'return_percentage': (total_gain_loss / total_cost * 100) if total_cost > 0 else 0,
            'holdings': [
                {
                    'symbol': inv.symbol,
                    'shares': inv.shares,
                    'current_price': inv.current_price,
                    'current_value': inv.current_value,
                    'gain_loss': inv.gain_loss
                }
                for inv in self.investments
            ]
        }
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive financial summary report"""
        report = []
        report.append("=" * 60)
        report.append("PERSONAL FINANCE DASHBOARD - SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current month summary
        current_date = datetime.now()
        current_month = self.calculate_monthly_summary(current_date.year, current_date.month)
        
        report.append("CURRENT MONTH SUMMARY")
        report.append("-" * 30)
        report.append(f"Income: ${current_month['income']:,.2f}")
        report.append(f"Expenses: ${current_month['expenses']:,.2f}")
        report.append(f"Net Income: ${current_month['net_income']:,.2f}")
        report.append(f"Transactions: {current_month['transaction_count']}")
        report.append("")
        
        # Expense breakdown
        if current_month['category_breakdown']:
            report.append("EXPENSE BREAKDOWN")
            report.append("-" * 20)
            for category, amount in sorted(current_month['category_breakdown'].items(), 
                                         key=lambda x: x[1], reverse=True):
                report.append(f"{category}: ${amount:,.2f}")
            report.append("")
        
        # Portfolio summary
        if self.investments:
            portfolio = self.get_portfolio_summary()
            report.append("INVESTMENT PORTFOLIO")
            report.append("-" * 20)
            report.append(f"Total Value: ${portfolio['total_value']:,.2f}")
            report.append(f"Total Cost: ${portfolio['total_cost']:,.2f}")
            report.append(f"Gain/Loss: ${portfolio['total_gain_loss']:,.2f}")
            report.append(f"Return: {portfolio['return_percentage']:.2f}%")
            report.append("")
            
            report.append("HOLDINGS")
            report.append("-" * 10)
            for holding in portfolio['holdings']:
                report.append(f"{holding['symbol']}: {holding['shares']} shares @ ${holding['current_price']:.2f} = ${holding['current_value']:,.2f}")
            report.append("")
        
        # Net worth trend
        if self.net_worth_history:
            latest_net_worth = self.net_worth_history[-1][1]
            report.append("NET WORTH")
            report.append("-" * 10)
            report.append(f"Current Net Worth: ${latest_net_worth:,.2f}")
            
            if len(self.net_worth_history) > 1:
                previous_net_worth = self.net_worth_history[-2][1]
                change = latest_net_worth - previous_net_worth
                report.append(f"Change from last period: ${change:,.2f}")
            report.append("")
        
        return "\n".join(report)
    
    def plot_financial_metrics(self) -> None:
        """Create visualizations of key financial metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Personal Finance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Monthly Income vs Expenses (last 6 months)
        months = []
        income_data = []
        expense_data = []
        
        for i in range(6):
            date = datetime.now() - timedelta(days=30*i)
            month_data = self.calculate_monthly_summary(date.year, date.month)
            months.append(date.strftime('%Y-%m'))
            income_data.append(month_data['income'])
            expense_data.append(month_data['expenses'])
        
        months.reverse()
        income_data.reverse()
        expense_data.reverse()
        
        ax1.bar([x + 0.2 for x in range(len(months))], income_data, 0.4, label='Income', color='green', alpha=0.7)
        ax1.bar([x - 0.2 for x in range(len(months))], expense_data, 0.4, label='Expenses', color='red', alpha=0.7)
        ax1.set_title('Monthly Income vs Expenses')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Amount ($)')
        ax1.set_xticks(range(len(months)))
        ax1.set_xticklabels(months, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Expense Categories (current month)
        current_month = self.calculate_monthly_summary(datetime.now().year, datetime.now().month)
        if current_month['category_breakdown']:
            categories = list(current_month['category_breakdown'].keys())
            amounts = list(current_month['category_breakdown'].values())
            
            ax2.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Expense Categories (Current Month)')
        else:
            ax2.text(0.5, 0.5, 'No expense data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Expense Categories')
        
        # 3. Net Worth Over Time
        if self.net_worth_history:
            dates = [datetime.strptime(entry[0], '%Y-%m-%d') for entry in self.net_worth_history]
            net_worths = [entry[1] for entry in self.net_worth_history]
            
            ax3.plot(dates, net_worths, marker='o', linewidth=2, markersize=6)
            ax3.set_title('Net Worth Over Time')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Net Worth ($)')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No net worth data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Net Worth Over Time')
        
        # 4. Investment Performance
        if self.investments:
            portfolio = self.get_portfolio_summary()
            symbols = [holding['symbol'] for holding in portfolio['holdings']]
            values = [holding['current_value'] for holding in portfolio['holdings']]
            
            ax4.bar(symbols, values, color='skyblue', alpha=0.7)
            ax4.set_title('Investment Holdings Value')
            ax4.set_xlabel('Symbol')
            ax4.set_ylabel('Value ($)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax4.text(i, v + max(values) * 0.01, f'${v:,.0f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No investment data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Investment Holdings Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'financial_dashboard.png'), dpi=300, bbox_inches='tight')
        print(f"Dashboard visualization saved to {os.path.join(self.data_dir, 'financial_dashboard.png')}")
        plt.show()


def main():
    """Main function to run the Personal Finance Dashboard"""
    print("Personal Finance Dashboard")
    print("=" * 40)
    
    # Initialize dashboard
    dashboard = PersonalFinanceDashboard()
    
    # Load data files
    print("\nLoading data...")
    dashboard.load_transactions('transactions.csv')
    dashboard.load_net_worth_history('net_worth.json')
    dashboard.load_investments('investments.json')
    
    # Generate and display summary report
    print("\n" + dashboard.generate_summary_report())
    
    # Create visualizations
    print("\nGenerating visualizations...")
    dashboard.plot_financial_metrics()
    
    print("\nDashboard complete!")


if __name__ == "__main__":
    main()