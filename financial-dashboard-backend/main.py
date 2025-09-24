"""
Financial Dashboard Backend API
A robust FastAPI application for managing financial transactions and portfolio holdings.
"""

from financial_logic import calculate_net_worth_timeline, calculate_asset_allocation
from datetime import datetime
from typing import List, Optional
import asyncio
import logging

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_URL = "sqlite:///./finances.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Transaction(Base):
    """
    SQLAlchemy model for financial transactions.
    
    Attributes:
        id: Primary key identifier
        date: Transaction timestamp
        amount: Transaction amount (positive for income, negative for expenses)
        category: Transaction category (e.g., 'Food', 'Salary', 'Investment')
        description: Transaction description
    """
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(String(255), nullable=False)


class Portfolio(Base):
    """
    SQLAlchemy model for portfolio holdings.
    
    Attributes:
        id: Primary key identifier
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        shares: Number of shares owned
    """
    __tablename__ = "portfolio"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, unique=True)
    shares = Column(Float, nullable=False)


# Pydantic Models for Request/Response Validation
class TransactionBase(BaseModel):
    """Base Pydantic model for transaction data validation."""
    amount: float = Field(..., description="Transaction amount")
    category: str = Field(..., min_length=1, max_length=100, description="Transaction category")
    description: str = Field(..., min_length=1, max_length=255, description="Transaction description")
    date: Optional[datetime] = Field(default=None, description="Transaction date (defaults to current time)")

    @validator('amount')
    def validate_amount(cls, v):
        if v == 0:
            raise ValueError('Amount cannot be zero')
        return v


class TransactionCreate(TransactionBase):
    """Pydantic model for creating new transactions."""
    pass


class TransactionResponse(TransactionBase):
    """Pydantic model for transaction API responses."""
    id: int
    date: datetime
    
    class Config:
        from_attributes = True


class PortfolioBase(BaseModel):
    """Base Pydantic model for portfolio data validation."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    shares: float = Field(..., gt=0, description="Number of shares (must be positive)")

    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()


class PortfolioCreate(PortfolioBase):
    """Pydantic model for creating/updating portfolio holdings."""
    pass


class PortfolioResponse(PortfolioBase):
    """Pydantic model for portfolio API responses."""
    id: int
    
    class Config:
        from_attributes = True


class PortfolioValueResponse(BaseModel):
    """Pydantic model for portfolio value calculation response."""
    total_value: float = Field(..., description="Total portfolio value in USD")
    holdings: List[dict] = Field(..., description="Individual holding values")


class MarketDataResponse(BaseModel):
    """Pydantic model for market data response."""
    ticker: str
    price: float
    timestamp: datetime


# Analytics Response Models
class NetWorthTimelineResponse(BaseModel):
    """Pydantic model for net worth timeline response."""
    date: datetime
    cumulative_net_worth: float


class AssetAllocationResponse(BaseModel):
    """Pydantic model for asset allocation response."""
    ticker: str
    shares: float
    current_price: float
    total_value: float
    percentage: float


# Database Dependency
def get_db() -> Session: # type: ignore
    """
    Database dependency function for FastAPI dependency injection.
    
    Provides a database session that is properly closed after use.
    This ensures proper connection management and prevents memory leaks.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Async Market Data Function
async def fetch_market_data(ticker: str) -> MarketDataResponse:
    """
    Asynchronous function to fetch real-time market data for a stock ticker.
    
    In a production environment, this would make HTTP requests to a real
    market data API (e.g., Alpha Vantage, IEX Cloud, Yahoo Finance).
    For demonstration purposes, this function simulates API latency and
    returns mock data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        MarketDataResponse: Mock market data with current timestamp
        
    Raises:
        HTTPException: If ticker is invalid or API simulation fails
    """
    # Simulate API latency
    await asyncio.sleep(0.1)
    
    # Mock market data - in production, this would be real API calls
    mock_prices = {
        'AAPL': 175.43,
        'GOOGL': 2847.63,
        'MSFT': 378.85,
        'AMZN': 3342.88,
        'TSLA': 248.42,
        'NVDA': 875.28,
        'META': 331.05,
        'NFLX': 445.87
    }
    
    ticker = ticker.upper()
    if ticker not in mock_prices:
        # For unknown tickers, generate a random price between $10-$500
        import random
        price = round(random.uniform(10.0, 500.0), 2)
    else:
        price = mock_prices[ticker]
    
    logger.info(f"Fetched market data for {ticker}: ${price}")
    
    return MarketDataResponse(
        ticker=ticker,
        price=price,
        timestamp=datetime.utcnow()
    )


# FastAPI Application
app = FastAPI(
    title="Financial Dashboard API",
    description="A robust backend API for managing financial transactions and portfolio holdings",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on application startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


# API Endpoints

@app.get("/api/transactions", response_model=List[TransactionResponse])
async def get_transactions(db: Session = Depends(get_db)):
    """
    Retrieve all transactions from the database.
    
    Returns:
        List[TransactionResponse]: List of all transactions ordered by date (newest first)
    """
    try:
        transactions = db.query(Transaction).order_by(Transaction.date.desc()).all()
        logger.info(f"Retrieved {len(transactions)} transactions")
        return transactions
    except Exception as e:
        logger.error(f"Error retrieving transactions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transactions"
        )


@app.post("/api/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(transaction: TransactionCreate, db: Session = Depends(get_db)):
    """
    Create a new transaction in the database.
    
    Args:
        transaction: Transaction data to be created
        
    Returns:
        TransactionResponse: The created transaction with assigned ID
    """
    try:
        # Set default date if not provided
        transaction_data = transaction.dict()
        if transaction_data['date'] is None:
            transaction_data['date'] = datetime.utcnow()
        
        db_transaction = Transaction(**transaction_data)
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        
        logger.info(f"Created transaction: {db_transaction.id} - {db_transaction.description}")
        return db_transaction
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating transaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create transaction"
        )


@app.get("/api/portfolio", response_model=List[PortfolioResponse])
async def get_portfolio(db: Session = Depends(get_db)):
    """
    Retrieve all portfolio holdings from the database.
    
    Returns:
        List[PortfolioResponse]: List of all portfolio holdings ordered by ticker
    """
    try:
        holdings = db.query(Portfolio).order_by(Portfolio.ticker).all()
        logger.info(f"Retrieved {len(holdings)} portfolio holdings")
        return holdings
    except Exception as e:
        logger.error(f"Error retrieving portfolio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio"
        )


@app.post("/api/portfolio", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_holding(holding: PortfolioCreate, db: Session = Depends(get_db)):
    """
    Create a new holding or update existing holding in the portfolio.
    
    If a holding with the same ticker already exists, it will be updated.
    Otherwise, a new holding will be created.
    
    Args:
        holding: Portfolio holding data
        
    Returns:
        PortfolioResponse: The created or updated holding
    """
    try:
        # Check if holding already exists
        existing_holding = db.query(Portfolio).filter(Portfolio.ticker == holding.ticker).first()
        
        if existing_holding:
            # Update existing holding
            existing_holding.shares = holding.shares
            db.commit()
            db.refresh(existing_holding)
            logger.info(f"Updated holding: {existing_holding.ticker} - {existing_holding.shares} shares")
            return existing_holding
        else:
            # Create new holding
            db_holding = Portfolio(**holding.dict())
            db.add(db_holding)
            db.commit()
            db.refresh(db_holding)
            logger.info(f"Created holding: {db_holding.ticker} - {db_holding.shares} shares")
            return db_holding
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating holding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create or update holding"
        )


@app.get("/api/portfolio/value", response_model=PortfolioValueResponse)
async def get_portfolio_value(db: Session = Depends(get_db)):
    """
    Calculate and return the total real-time value of the portfolio.
    
    This endpoint combines portfolio holdings from the database with
    real-time market data to calculate the current portfolio value.
    Uses asynchronous operations to fetch market data efficiently.
    
    Returns:
        PortfolioValueResponse: Total portfolio value and individual holding details
    """
    try:
        # Get all portfolio holdings
        holdings = db.query(Portfolio).all()
        
        if not holdings:
            return PortfolioValueResponse(total_value=0.0, holdings=[])
        
        # Fetch market data for all holdings concurrently using async operations
        market_data_tasks = [fetch_market_data(holding.ticker) for holding in holdings]
        market_data_list = await asyncio.gather(*market_data_tasks)
        
        # Calculate total value and prepare response
        total_value = 0.0
        holding_values = []
        
        for holding, market_data in zip(holdings, market_data_list):
            holding_value = holding.shares * market_data.price
            total_value += holding_value
            
            holding_values.append({
                'ticker': holding.ticker,
                'shares': holding.shares,
                'price_per_share': market_data.price,
                'total_value': holding_value,
                'last_updated': market_data.timestamp.isoformat()
            })
        
        logger.info(f"Calculated portfolio value: ${total_value:.2f} across {len(holdings)} holdings")
        
        return PortfolioValueResponse(
            total_value=round(total_value, 2),
            holdings=holding_values
        )
    except Exception as e:
        logger.error(f"Error calculating portfolio value: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate portfolio value"
        )


# Analytics Endpoints using Financial Logic Module
@app.get("/api/analytics/net-worth-timeline", response_model=List[NetWorthTimelineResponse])
async def get_net_worth_timeline(db: Session = Depends(get_db)):
    """
    Calculate and return net worth timeline based on transaction history.
    
    This endpoint uses the financial logic module to compute cumulative
    net worth over time, showing how wealth changes with each transaction.
    
    Returns:
        List[NetWorthTimelineResponse]: Timeline of cumulative net worth
    """
    try:
        transactions = db.query(Transaction).all()
        
        if not transactions:
            logger.info("No transactions found for net worth timeline")
            return []
        
        transaction_data = [
            {
                'date': t.date,
                'amount': t.amount,
                'category': t.category,
                'description': t.description
            }
            for t in transactions
        ]
        
        timeline_df = calculate_net_worth_timeline(transaction_data)
        logger.info(f"Generated net worth timeline with {len(timeline_df)} data points")
        
        return timeline_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error generating net worth timeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate net worth timeline"
        )


@app.get("/api/analytics/asset-allocation", response_model=List[AssetAllocationResponse])
async def get_asset_allocation(db: Session = Depends(get_db)):
    """
    Calculate and return current asset allocation percentages.
    
    This endpoint combines portfolio holdings with current market prices
    to show what percentage each holding represents of the total portfolio.
    
    Returns:
        List[AssetAllocationResponse]: Asset allocation breakdown
    """
    try:
        holdings = db.query(Portfolio).all()
        
        if not holdings:
            logger.info("No portfolio holdings found for asset allocation")
            return []
        
        # Fetch current prices for all holdings
        market_data_tasks = [fetch_market_data(holding.ticker) for holding in holdings]
        market_data_list = await asyncio.gather(*market_data_tasks)
        
        # Create price dictionary
        current_prices = {
            market_data.ticker: market_data.price 
            for market_data in market_data_list
        }
        
        # Prepare holdings data
        holdings_data = [
            {
                'ticker': holding.ticker,
                'shares': holding.shares
            }
            for holding in holdings
        ]
        
        allocation_df = calculate_asset_allocation(holdings_data, current_prices)
        logger.info(f"Generated asset allocation for {len(allocation_df)} holdings")
        
        return allocation_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error generating asset allocation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate asset allocation"
        )


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Simple health status message
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    # Run the application using uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )