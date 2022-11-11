import yfinance as yf
import pandas as pd
import numpy as np
import math
from scipy.optimize import LinearConstraint, Bounds, minimize


def mark_obj(weights, cov):
    output = 1 / 2 * np.matmul(weights, cov)
    output = np.matmul(output, weights.transpose())
    return float(output)


class Equities:
    def __init__(self, stocks, start, end):
        self.stocks = stocks
        self.start = start
        self.end = end

        prices = yf.download(stocks, start, end)
        prices = prices['Adj Close']
        self.prices = prices

        log_returns = pd.DataFrame()
        equities = []
        for stock in prices.columns:
            log_returns[stock] = np.log(prices[stock]) - np.log(prices[stock].shift(1))
            equities.append(stock)
        self.log_returns = log_returns
        self.equities = equities

    def daily_returns(self):
        output = []
        for stock in self.log_returns.columns:
            output.append(self.log_returns[stock].mean())
        return output

    def annual_returns(self):
        output = []
        for stock in self.log_returns.columns:
            output.append(self.log_returns[stock].mean()*252)
        output = pd.DataFrame(output)
        output.index = self.equities
        return output

    def max_ind_annual_return(self):
        output = float(self.annual_returns().max(0))
        return output

    def min_ind_annual_return(self):
        output = float(self.annual_returns().min(0))
        return output

    def cov_matrix(self):
        output = self.log_returns.cov()
        return output

    def mark_portfolio(self, target):
        returns = self.daily_returns()
        cov = self.cov_matrix()
        w_0 = np.zeros(len(returns))
        w_0 += 0.1

        # Bound weights in individual assets on [-1, 1]
        bounds = Bounds(-1*np.ones(len(returns)), np.ones(len(returns)))

        # Define constraints for optimization:
        # 1. Fully Invested (Sum of all holdings is 1)
        # 2. Expected Portfolio Return equals target (1.5%)
        cons = np.vstack((np.ones(len(returns)), returns))
        lb = [1, target/252]
        ub = [1, target/252]
        linear_constraint = LinearConstraint(cons, lb, ub)

        opt = minimize(mark_obj, w_0, method='trust-constr', args=cov,
                       constraints=[linear_constraint], bounds=bounds)

        output = pd.DataFrame(opt.x)
        output.index = self.equities

        return output


class Portfolio():
    def __init__(self, weights, returns, cov):
        self.weights = weights
        self.returns = returns
        self.cov = cov

    def port_return(self):
        output = self.weights.multiply(self.returns, axis=0)
        output = float(np.sum(output)*252)
        output = round(output, 4)
        return output

    def port_leverage(self):
        leverage = float(np.sum(self.weights.__abs__()))
        leverage = round(leverage, 4)
        return leverage

    def port_risk(self):
        temp = self.weights.T.dot(self.cov)
        temp2 = temp.dot(self.weights)
        temp2 = round(math.sqrt(temp2[0]*252), 4)

        return temp2


if __name__ == '__main__':
    universe = Equities(['AAPL', 'MSFT', 'TSLA', 'GS', 'BA', 'VZ', 'XOM',
                         'MMM', 'GE', 'WMT'], '2021-1-1', '2021-4-1')

    holdings = universe.mark_portfolio(target=0.5)
    leverage = holdings.__abs__()
    leverage = np.sum(leverage)

    exp_rets = universe.daily_returns()
    cov_mtrx = universe.cov_matrix()

    print('ANNUAL RETURNS')
    print(universe.annual_returns())
    print('')

    print('PORTFOLIO HOLDINGS:')
    print(holdings)
    print('')

    mark_port = Portfolio(holdings, exp_rets, cov_mtrx)
    print('PORTFOLIO RETURN')
    print(mark_port.port_return())
    print('')

    print('PORTFOLIO LEVERAGE')
    print(mark_port.port_leverage())
    print('')

    print('PORTFOLIO RISK')
    print(mark_port.port_risk())








